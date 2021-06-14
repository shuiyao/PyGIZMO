'''
The accretion tracking engine.
The engine will find all galactic gas accretion events during a given time and
sort them into different categories according to the histories of the accreted 
gas particles. Three primary categories are cold accretion, hot accretion and 
wind re-accretion. The third category, in particular, requires tracking when 
and where the gas particles gain their mass from galactic winds and is most 
challenging.

Test case for the wind tracking engine
--------------------------------
model: l25n144-test
snapnum: 108
HaloId = 1185:
  + Isolated galaxy
  + satellites (1219, 1086)
HaloId = 715:
  + In a dense environment
  + Many satellites
HaloId = 568:
  + In a cluster environment
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
import simulation

schema_gptable = {'columns':['PId','snapnum','Mass','haloId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'Mass':'float32',
                            'haloId':'int32'}
}
schema_pptable = {'columns':['PId','haloId','snapnum','Mloss','snapfirst','birthId'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'haloId':'int32',
                            'Mloss':'float64',
                            'snapfirst':'int32',                            
                            'birthId':'int32'}
}

# gptable can be temporary, unless we can handle a gigantic table with Spark

# Parquet I/O
#https://blog.datasyndrome.com/python-and-parquet-performance-e71da65269ce

model = "l25n144-test"

def build_gptable(pidlist, snaplast, snapstart=0, overwrite=False):
    '''
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

def build_pptable(gptable, inittable, phewtable, overwrite=False):
    '''
    Build or load a temporary table that stores all the necessary attributes 
    of selected PhEW particles that can be queried by the accretion tracking 
    engine.

    Parameters
    ----------
    gptable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId, ...
        Gas particles whose histories we are tracking.
        Needed to find the list of halos for processing
    inittable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
        The initial/final status of all PhEW particles
        Needed to know from which halo a PhEW particles were born
        Created from Simulation.build_inittable_from_simulation()
    phewtable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId, (Mloss)
        All PhEW particles in any snapshot of the simulation
        The function return is a subset of it
        Created from Simulation.build_phewtable_from_simulation()

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
    halos = compile_halos_to_process(gptable)

    # Find all PhEW particles that ever appeared in these halos
    pptable = pd.merge(halos, phewtable, how='inner',
                       left_on=['snapnum', 'haloId'],
                       right_on=['snapnum', 'haloId'])
    # pptable: snapnum, haloId, PId, Mloss
    
    # Add the birth halo information to pptable
    pptable = pd.merge(pptable, inittable[['PId','snapfirst','birthId']],
                       how='left', left_on='PId', right_on='PId')

    schema = utils.pyarrow_read_schema(schema_pptable)
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

def assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap):
    '''
    Map from the unique halo identifier (snapnum, haloId) to a descriptor for 
    its relationship with another halo (haloIdTarget) at a later time.
    In the accretion tracking engine, the table is built iteratively for each 
    halo of interest.

    Parameters
    ----------
    haloIdTarget: int.
        The haloId of the halo in the current snapshot.
    halos: pandas.DataFrame
        Columns: haloId*, snapnum*
        Halos uniquely identified with the (haloId, snapnum) pair
    progtable: pandas.DataFrame.
        Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
        Output of progen.find_all_previous_progenitors().
        Defines the progenitors of any halo in any previous snapshot.
    hostmap: pandas.DataFrame.
        Columns: snapnum*, haloId*, hostId
        Output of progen.build_haloId_hostId_map()
        Mapping between haloId and hostId for each snapshot.

    Returns:
    relation: pandas.DataFrame
        Columns: snapnum*, haloId*, relation
        Map between a halo (MultiIndex(snapnum, haloId)) to the relation, which 
        defines its relation to another halo (haloIdTarget) at a later time.
    '''

    progsTarget = progtable.loc[progtable.index == haloIdTarget,
                                ['snapnum', 'progId', 'hostId']]
    progsTarget.rename(columns={'hostId':'progHost'}, inplace=True)
    
    halos = halos[['snapnum', 'haloId']].set_index(['snapnum', 'haloId'])

    # Find the hostId of each halo in its snapshot
    halos = halos.join(hostmap, how='left') # (snapnum, haloId) -> hostId
    halos = halos.reset_index()

    halos = pd.merge(halos, progsTarget, how='left',
                     left_on = 'snapnum', right_on = 'snapnum')
    halos['relation'] = halos.apply(lambda x : define_halo_relationship(
        x.progId, x.progHost, x.haloId, x.hostId), axis=1)
    
    return halos[['snapnum','haloId','relation']].set_index(['snapnum', 'haloId'])

def add_relation_field_to_gptable(haloIdTarget, gptable, progtable, hostmap):
    '''
    Add a field ('relation') in the gptable that defines the relation between 
    haloIdTarget and any halo that hosts a gas particle in the gptable.

    Parameters
    ----------
    haloIdTarget: int.
        The haloId of the halo in the current snapshot.
    gptable: pandas.DataFrame or Spark DataFrame
    progtable: pandas.DataFrame.
        Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
        Output of progen.find_all_previous_progenitors().
        Defines the progenitors of any halo in any previous snapshot.
    hostmap: pandas.DataFrame.
        Columns: snapnum*, haloId*, hostId
        Output of progen.build_haloId_hostId_map()
        Mapping between haloId and hostId for each snapshot.

    Returns:
    gptable: pandas.DataFrame
    '''

    halos = compile_halos_to_process(gptable, ['snapnum','haloId'])
    halos = assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
    gptable = pd.merge(gptable, halos, how='left',
                       left_on=['snapnum','haloId'], right_index=True)
    gptable['relation'] = gptable['relation'].fillna('IGM')
    return gptable

def add_birthtag_field_to_pptable(haloIdTarget, pptable, progtable, hostmap):
    '''
    Add a field ('birthTag') in the pptable that defines the relation between 
    haloIdTarget and the halo where a PhEW particle was born.

    Parameters
    ----------
    haloIdTarget: int.
        The haloId of the halo in the current snapshot.
    pptable: pandas.DataFrame or Spark DataFrame
        Columns: PId*, snapnum, Mass, haloId, snapfirst, birthId
        PhEW particles that shared mass with the gas particles
    progtable: pandas.DataFrame.
        Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
        Output of progen.find_all_previous_progenitors().
        Defines the progenitors of any halo in any previous snapshot.
    hostmap: pandas.DataFrame.
        Columns: snapnum*, haloId*, hostId
        Output of progen.build_haloId_hostId_map()
        Mapping between haloId and hostId for each snapshot.

    Returns:
    pptable: pandas.DataFrame
    '''

    halos = compile_halos_to_process(pptable, ['snapfirst','birthId'])
    halos = assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
    pptable = pd.merge(pptable, halos, how='left',
                       left_on=['snapfirst','birthId'], right_index=True)
    pptable.rename(columns={'relation':'birthTag'}, inplace=True)
    return pptable


def fetch_phew_particles_for_halos(snap, hids):
    '''
    Fetch all PhEW particles in a snapshot that appear in any of the halo in 
    a list of halos.

    Parameter
    ---------
    snap: class Snapshot.
    hids: array type.
        A list of haloIds in the snapshot.

    Returns
    -------
    phewp: pandas.DataFrame.
        Columns: haloId, snapnum, PId.
        A list of PhEW partcles that appear in the halos.

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

def compile_halos_to_process(gptable, fields=['snapnum', 'haloId']):
    halos = gptable.loc[:, fields]
    halos.rename(columns={fields[0]:'snapnum', fields[1]:'haloId'}, inplace=True)
    halos = halos[halos.haloId != 0].drop_duplicates()
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

REFRESH = False

sim = simulation.Simulation(model)
snapnum = 108
snap = snapshot.Snapshot(model, snapnum)

# Find progenitors for all halos within a snapshot.
progtable = find_all_previous_progenitors(snap, overwrite=REFRESH)
hostmap = build_haloId_hostId_map(snap, overwrite=REFRESH)

# Time-consuming, but only needs one-go for each simulation
inittable = sim.build_inittable(overwrite=False)
sim.build_phewtable(overwrite=False)
sim.compute_mloss_partition_by_pId(overwrite=False)

phewtable = sim.load_phewtable()

# From here on, galaxy level
# Get all gas particles from the target galaxy
haloIdTarget = 1185 # Npart = 28, logMgal = 9.5, within 0.8-0.9 range

# Want to do a couple of things:
# 1a. Find all gas particles currently in a galaxy
# 1b. Find all recent accretion onto the galaxy
# 2. Classify the primodial component into COLD and HOT
# 3. Get the wind accretion info (SELF, SAT, IGM, ...)
#   - Diagnostic: For each gas particle, plot Mgain vs. snapnum
#     - Experiment with wacc.1185.csv
# 4. Add up all COLD, HOT, SELF, SAT, IGM for the galaxy
# Another track:
# Find the mass fraction in SELF, CEN, SAT, IGM as a function of time.
# Just use the gptable and find curtag


pidlist = snap.get_gas_particles_in_galaxy(galIdTarget)

REFRESH = False
# Build gptable for selected gas particles (gId=1185;01:21)
gptable = build_gptable(pidlist, snapnum, 0, overwrite=REFRESH)
gptable = sim.compute_mgain_partition_by_pId(gptable)
gptable = add_relation_field_to_gptable(galIdTarget, gptable, progtable, hostmap)
# gptable.haloId = gptable.haloId.apply(abs)

pptable = build_pptable(gptable, inittable, phewtable, overwrite=REFRESH)
pptable = add_birthtag_field_to_pptable(galIdTarget, pptable, progtable, hostmap)

# This is only for code test
# pptable.loc[(pptable.snapfirst==30)&(pptable.birthId==2), 'birthTag'] = 'SAT'

def compute_wind_mass_partition_by_birthtag(gptable, pptable):
    '''
    The final step of the wind tracking engine. Compute how much wind material
    a gas particle gained during each snapshot. The gained wind material is 
    divided into categories based on the relation between a target halo and 
    the halo from which the wind particle was launched.

    Returns
    -------
    mwindtable: pandas.DataFrame
        Columns: PId*, snapnum, birthTag, Mgain
    '''
    grps = pptable.groupby(['snapnum', 'haloId'])
    x = grps.apply(lambda x: x[['Mloss','birthTag']].groupby('birthTag').sum()).reset_index('birthTag')
    y = pd.merge(gptable, x, how='left', left_on=['snapnum', 'haloId'], right_on=['snapnum', 'haloId'])
    grps = y.groupby(['PId','snapnum'])
    mwindtable = grps.apply(lambda x : pd.DataFrame({
        'PId': x.PId,
        'snapnum': x.snapnum,
        'birthTag':x.birthTag,
        'Mgain':x.Mgain * x.Mloss / x.Mloss.sum()})
    )
    # mwindtable = z.groupby(['PId', 'birthTag']).sum()
    return mwindtable
