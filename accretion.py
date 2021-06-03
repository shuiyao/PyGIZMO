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

import pdb

schema_gptable = {'columns':['PId','snapnum','Mass','haloId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'Mass':'float32',
                            'haloId':'int32'}
}
schema_pptable = {'columns':['haloId','snapnum','PId'],
                  'dtypes':{'haloId':'int32',
                            'snapnum':'int32',
                            'PId':'int64'}
}

# TODO: Build the gptable and inittable
# gptable can be temporary, unless we can handle a gigantic table with Spark

# Procedure:
# 1a. [DONE] Identify the gas particles we want to track.
#    - Want to know gp.mwind = {'SELF':, 'SAT':, ...}
# 1b. [DONE] Create gptable (n_gp * n_snapnum)
# 2. [DONE] Track each of them back in time, compile a list of halos to do. (haloId, snapnum)
# 3a. [DONE] For each (halo, snapnum), get all PhEWs in it
# 3b. Assign to each PhEW, snapnum_init, haloId_init, host_status (['SELF', 'SAT', ...])
# 3c. For each (halo, snapnum), compute mgain = {'SELF':, 'SAT'}
# 4. Track gp through time again, accumulating dmass (x) (halo, snapnum).mgain

# Parquet I/O
#https://blog.datasyndrome.com/python-and-parquet-performance-e71da65269ce

def fetch_phew_particles_for_halos(snap, hids):
    '''
    >>> hids = pd.Series([534, 584, 374])
    '''
    if(isinstance(hids, list)): hids = pd.Series(hids)
    hids = hids[hids>0]
    snap.load_gas_particles(['PId','haloId','Mc'], drop=False)
    pid = hids.apply(lambda x : list(snap.gp.query("haloId==@x and Mc>0").PId))
    phewp = pd.DataFrame({'haloId':hids, 'snapnum':snap.snapnum, 'PId':pid})
    # Need to remove halos that do not have any PhEW particles
    phewp = phewp.explode('PId').dropna()
    talk("{} PhEW particles fetched for {} halos.".format(phewp.shape[0], hids.size), 'talky')
    return phewp

pidlist = [200000, 200001, 300000, 300001]
def load_gptable_and_pptable(pidlist, snaplast, snapstart=0, overwrite=False):
    '''
    >>> pidlist = [200000, 200001, 300000, 300001]
    >>> gptable, pptable = load_gptable_and_pptable(pidlist, 190, 200)
    '''
    path_gptable = os.path.join(cfg['Paths']['tmp'], "gptable.parquet")
    path_pptable = os.path.join(cfg['Paths']['tmp'], "pptable.parquet")    
    if(overwrite==False and os.path.exists(path_gptable)):
        talk("Load existing gptable and pptable", 'normal')
        gptable = pd.read_parquet(path_gptable)
        pptable = pd.read_parquet(path_pptable)
        return gptable, pptable
    talk("Building gptable for {} gas particles".format(len(pidlist)), 'normal')
    gptable, pptable = None, None
    for snapnum in tqdm(range(snapstart, snaplast+1), desc='snapnum', ascii=True):
        snap = snapshot.Snapshot('l12n144-phew-movie-200', snapnum)
        snap.load_gas_particles(['PId','Mass','haloId','Mc'])
        gp = snap.gp.query('PId in @pidlist')

        # Here we know which halos from this snapshot needs to be processed in 
        # the future. Now we get all the PhEW particles that happen to be in 
        # these halos
        pp = fetch_phew_particles_for_halos(snap, gp.haloId.drop_duplicates())
        
        gp.loc[:,'snapnum'] = snapnum
        gptable = gp.copy() if gptable is None else pd.concat([gptable, gp])
        pptable = pp.copy() if pptable is None else pd.concat([pptable, pp])
        
    schema = utils.get_pyarrow_schema_from_json(schema_gptable)
    tab = pa.Table.from_pandas(gptable, schema=schema, preserve_index=False)
    pq.write_table(tab, path_gptable)
    schema = utils.get_pyarrow_schema_from_json(schema_pptable)
    tab = pa.Table.from_pandas(pptable, schema=schema, preserve_index=False)
    pq.write_table(tab, path_pptable)
    return gptable, pptable

def compile_halos_to_do(gptable):
    return gptable[['snapnum','haloId']].drop_duplicates()

# gptable, pptable = load_gptable(pidlist, 200, 190, overwrite=True)
# gptable, pptable = load_gptable(pidlist, 200, 190, overwrite=False)

# snap = snapshot.Snapshot('l12n144-phew-movie-200', 100)
# snap.load_gas_particles(['PId','haloId','Mc'])

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
