'''
The accretion tracking engine.
The engine will find all galactic gas accretion events during a given time and
sort them into different categories according to the histories of the accreted 
gas particles. Three primary categories are cold accretion, hot accretion and 
wind re-accretion. The third category, in particular, requires tracking when 
and where the gas particles gain their mass from galactic winds and is most 
challenging.
'''

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

# TODO: Tag the pptable (query progtable and hostmap)
# TODO: Aggregate

schema_gptable = {'columns':['PId','snapnum','Mass','haloId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'Mass':'float32',
                            'haloId':'int32'}
}
schema_pptable = {'columns':['PId','haloId','snapnum','Mloss','birthId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'haloId':'int32',
                            'Mloss':'float64',
                            'birthId':'int32'}
}

# gptable can be temporary, unless we can handle a gigantic table with Spark

# Parquet I/O
#https://blog.datasyndrome.com/python-and-parquet-performance-e71da65269ce

model = "l25n144-test"

pidlist = [707945, 728826, 667761, 604544]
def build_gptable(pidlist, snaplast, snapstart=0, overwrite=False):
    '''
    >>> pidlist = [200000, 200001, 300000, 300001]
    >>> gptable, pptable = load_gptable_and_pptable(pidlist, 190, 200)
    '''
    path_gptable = os.path.join(cfg['Paths']['tmp'], "gptable.parquet")
    if(overwrite==False and os.path.exists(path_gptable)):
        talk("Load existing gptable.", 'normal')
        gptable = pd.read_parquet(path_gptable)
        return gptable
    talk("Building gptable for {} gas particles".format(len(pidlist)), 'normal')
    gptable = None
    for snapnum in tqdm(range(snapstart, snaplast+1), desc='snapnum', ascii=True):
        snap = snapshot.Snapshot(model, snapnum)
        snap.load_gas_particles(['PId','Mass','haloId'])
        # gp = snap.gp.query('PId in @pidlist')
        gp = snap.gp.loc[snap.gp.PId.isin(pidlist), :]
        gp.loc[:,'snapnum'] = snapnum
        gptable = gp.copy() if gptable is None else pd.concat([gptable, gp])

    gptable = gptable.set_index('PId').sort_values('snapnum')
    schema = utils.pyarrow_read_schema(schema_gptable)
    tab = pa.Table.from_pandas(gptable, schema=schema, preserve_index=True)
    pq.write_table(tab, path_gptable)
    return gptable

def build_pptable(gptable, inittable, phewtable):
    '''
    Build or load a temporary table that stores all the necessary attributes 
    of selected PhEW particles that can be queried by the accretion tracking 
    engine.

    Parameters
    ----------
    gptable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId
        Gas particles whose histories we are tracking.
        Needed to find the list of halos for processing
    inittable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
        The initial/final status of all PhEW particles
        Needed to know from which halo a PhEW particles were born
    phewtable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId, (Mloss)
        All PhEW particles in any snapshot of the simulation
        The function return is a subset of it

    Returns
    -------
    pptable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, haloId, Mloss, birthId
        Temporary table storing all the necessary attributes of selected PhEW
        particles that can be queried by the accretion tracking engine.
    '''

    path_pptable = os.path.join(cfg['Paths']['tmp'], "pptable.parquet")
    if(overwrite==False and os.path.exists(path_pptable)):
        talk("Load existing pptable.", 'normal')
        pptable = pd.read_parquet(path_pptable)
        return pptable
    
    # Find all halos that ever hosted the gas particles in gptable
    halos_to_do = compile_halos_to_do(gptable)

    # Find all PhEW particles that ever appeared in these halos
    # Computionally intensive.
    pptable = pd.merge(gptable, phewtable, how='left',
                       left_on=['snapnum', 'haloId'],
                       right_on=['snapnum', 'haloId'])
    # pptable: snapnum, haloId, PId, Mloss
    
    # Add the birth halo information to pptable
    pptable = pd.merge(pptable, inittable[['snapfirst','birthId']], how='left',
                       left_on='PId', right_index=True)

    schema = utils.get_pyarrow_schema_from_json(schema_pptable)
    tab = pa.Table.from_pandas(pptable, schema=schema, preserve_index=False)
    pq.write_table(tab, path_pptable)
    return pptable

def define_halo_relationship(progId, progHost, haloId, hostId):
    if(progId == 0): return "IGM"
    if(progId == haloId): return "SELF"
    if(progHost == haloId): return "PARENT"
    if(progId == hostId): return "SAT"
    if(progHost == hostId): return "SIB"
    return "IGM"

def build(haloIdTarget, gptable, progtable, hostmap):
    '''
    Map from the unique halo identifier (snapnum, haloId) to a descriptor for 
    their relationship.

    Parameters
    ----------
    haloIdTarget: int.
        The haloId of the halo in the current snapshot.
    gptable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId
        Gas particles whose histories we are tracking.
        Needed to find the list of halos for processing
    progtable: pandas.DataFrame.
        Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
        Output of progen.find_all_previous_progenitors().
        Defines the progenitors of any halo in any previous snapshot.
    hostmap: pandas.DataFrame.
        Columns: snapnum*, haloId*, hostId
        Output of progen.build_haloId_hostId_map()
        Mapping between haloId and hostId for each snapshot.
    '''

    progsTarget = progtable.loc[progtable.haloId == haloIdTarget,
                                ['snapnum', 'progId', 'hostId']]
    progsTarget.rename(columns={'hostId':'progHost'}, inplace=True)
    
    halos = compile_halos_to_do(gptable)
    halos = halos[halos.snapnum < 52].set_index(['snapnum', 'haloId'])

    # Find the hostId of each halo in its snapshot
    halos = halos.join(hostmap, how='left') # (snapnum, haloId) -> hostId
    halos = halos.reset_index()

    halos = pd.merge(halos, progsTarget, how='left',
                     left_on = 'snapnum', right_on = 'snapnum')
    halos['birthTag'] = halos.apply(lambda x : define_halo_relationship(
        x.progId, x.progHost, x.haloId, x.hostId), axis=1)
    
    return halos[['snapnum','haloId','birthTag']].set_index(['snapnum', 'haloId'])

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
    halos = gptable[['snapnum','haloId']]
    halos = halos[halos.haloId > 0].drop_duplicates()
    return halos

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

import progen
from progen import *
snap = snapshot.Snapshot(model, 52)
gptable = build_gptable(pidlist, 52, 48, overwrite=False)
progtable = find_all_previous_progenitors(snap)
hostmap = build_haloId_hostId_map(snap)
x = build(4, gptable, progtable, hostmap)
