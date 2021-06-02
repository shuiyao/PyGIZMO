import numpy as np
import pandas as pd
import snapshot
from myinit import *
import utils
from utils import talk
from tqdm import tqdm

from config import cfg

import pyarrow as pa
import pyarrow.parquet as pq

schema_gptable = {'columns':['PId','snapnum','Mass','haloId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'Mass':'float32',
                            'haloId':'int32'}
}


# TODO: Build the gptable and inittable
# gptable can be temporary, unless we can handle a gigantic table with Spark

# Procedure:
# 1a. [DONE] Identify the gas particles we want to track.
#    - Want to know gp.mwind = {'SELF':, 'SAT':, ...}
# 1b. [DONE] Create gptable (n_gp * n_snapnum)
# 2. [DONE] Track each of them back in time, compile a list of halos to do. (haloId, snapnum)
# 3a. For each (halo, snapnum), get all PhEWs in it
# 3b. Assign to each PhEW, snapnum_init, haloId_init, host_status (['SELF', 'SAT', ...])
# 3c. For each (halo, snapnum), compute mgain = {'SELF':, 'SAT'}
# 4. Track gp through time again, accumulating dmass (x) (halo, snapnum).mgain

# Parquet I/O
#https://blog.datasyndrome.com/python-and-parquet-performance-e71da65269ce

pidlist = [200000, 200001, 300000, 300001]
def load_gptable(pidlist, snaplast, fprefix='gptable', overwrite=False):
    fname = os.path.join(cfg['Paths']['tmp'], fprefix+".parquet")
    if(overwrite==False and os.path.exists(fname)):
        talk("Load existing gptable", 'normal')
        return pd.read_parquet(fname)
    talk("Building gptable for {} gas particles".format(len(pidlist)), 'normal')
    gptable = None
    for snapnum in tqdm(range(0, snaplast+1), desc='snapnum', ascii=True):
        snap = snapshot.Snapshot('l12n144-phew-movie-200', snapnum)
        snap.load_gas_particles(['PId','Mass','haloId'])
        gp = snap.gp.query('PId in @pidlist')
        gp.loc[:,'snapnum'] = snapnum
        gptable = gp.copy() if gptable is None else pd.concat([gptable, gp])
    schema = utils.get_pyarrow_schema_from_json(schema_gptable)
    tab = pa.Table.from_pandas(gptable, schema=schema, preserve_index=False)
    pq.write_table(tab, fname)
    return gptable

def compile_halos_to_do(gptable):
    return gptable[['snapnum','haloId']].drop_duplicates()

def load_timeinfo_for_snapshots():
    df_redz = pd.read_csv("redshifts.txt", sep='\s+', header=0)
    df_redz = df_redz.drop("#snapnum", axis=1)
    return df_redz

gptable = load_gptable(pidlist, 20)

# Find the first snapshot after ainit
#snapnum_first = (df_redz.a >= ainit).argmax()

# Select all PhEW particles inside a given halo at snapnum
#gptable.query("snapnum==@snapnum AND hostId==@hostId AND Mc > 0")

def gp_mass_gain_since_last_snapshot(PId, snapnum, gptable):
    '''
    Find the amount of mass gained in a normal gas particle (PId) between the 
    previous snapshot (snapnum-1) and the current snapshot (snapnum)
    '''
    # TODO: What if the gas particle gets accreted?
    # TODO: What if the gas particle gets splitted?
    # TODO: What if the gas particle hasn't appeared in the previous snapshot?
    if(snapnum == 0): return 0.0
    mass_this = gptable.query('snapnum==@snapnum AND PId==@PId').Mass
    mass_last = gptable.query('snapnum==@snapnum-1 AND PId==@PId').Mass
    return mass_this - mass_last

def phew_mass_loss_since_last_checkpoint(PId, snapnum, gptable, inittable):
    '''
    Find the amount of mass loss for a PhEW particle (PID) between the 
    last checkpoint MAX(a{snapnum-1}, ainit). 

    Parameters
    ----------
    PId: int
        The Particle ID of the PhEW particle
    snapnum: int
        The end time for query.
    gptable: pandas.DataFrame.
        The table for all gas particles
        Columns: PId*, snapnum, Mass, haloId, ...
    inittable: pandas.DataFrame.
        The table for the initwinds information
        Columns: PId*, ainit, minit

    Return
    ------
    Mass loss of the PhEW particle during the given period of time.
    '''

    mass_this = gptable.query('snapnum==@snapnum AND PId==@PId').Mass
    ainit = float(inittable.query('PId==@PId').ainit)
    minit = float(inittable.query('PId==@PId').minit)
    if(redzTable.loc[snapnum-1].a < ainit):
        mass_last = minit
    else:
        mass_last = gptable.query('snapnum==@snapnum-1 AND PId==@PId').Mass
    return mass_this - mass_last

# Mass Change From Last Checkpoint
# 1. Normal Gas Particle: dMass = Mwind[i] - Mwind[i-1] or Mwind[acc] - Mwind[i-1]
#    (snapnum, PID) -> Mgain
# 2. PhEW Particle: dMass = Mass[i] - Mass[i-1] or Mass[i] - Mass[init]
#    (snapnum, PID) -> Mloss
