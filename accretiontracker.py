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

class AccretionTracker():
    '''
    Class that deals with accretion tracking for galaxies in a single snapshot.

    Parameters
    ----------
    snap: Class Snapshot.
        The snapshot which to track accretion
    '''
    
    def __init__(self, snap):
        self._snap = snap
        self._simulation = snap._simulation
        self.model = snap.model
        self.snapnum = snap.snapnum
        self.path_base = self._simulation._path_tmpdir
        self.gptable = None
        self.pptable = None

    def initialize(self):
        # prepare all required permanent tables for accretion tracking. Build the tables if they are not yet existed.
        self._simulation.build_inittable()
        self._simulation.load_phewtable()
        # snap._simulation.compute_mloss_partition_by_pId(overwrite=False)
        self._simulation.load_hostmap()
        self._snap.load_progtable()

    def build_gptable(self, pidlist, snapstart=0, overwrite=False):
        '''
        Build the gptable for the particles to track. The particles are 
        specified by their particle IDs in the pidlist.
        '''

        snaplast = self.snapnum
        
        path_gptable = os.path.join(self.path_base, "gptable.parquet")
        if(overwrite==False and os.path.exists(path_gptable)):
            talk("Load existing gptable.", 'normal')
            gptable = pd.read_parquet(path_gptable)
            self.gptable = gptable
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
        self.gptable = gptable
        return gptable

    def build_pptable(self, inittable, phewtable, gptable=None, overwrite=False):
        '''
        Build or load a temporary table that stores all the necessary attributes 
        of selected PhEW particles that can be queried by the accretion tracking 
        engine.

        Parameters
        ----------
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
        gptable: pandas.DataFrame or Spark DataFrame. Default=None
            Columns: PId*, snapnum, Mass, haloId, ...
            Gas particles whose histories we are tracking.
            Needed to find the list of halos for processing
            If None, try self.gptable

        Returns
        -------
        pptable: pandas.DataFrame or Spark DataFrame
            Columns: PId*, snapnum, haloId, Mloss, birthId
            Temporary table storing all the necessary attributes of selected PhEW
            particles that can be queried by the accretion tracking engine.
        '''
        
        path_pptable = os.path.join(self.path_base, "pptable.parquet")
        if(overwrite==False and os.path.exists(path_pptable)):
            talk("Load existing pptable.", 'normal')
            pptable = pd.read_parquet(path_pptable)
            self.pptable = pptable
            return pptable

        if(gptable is None):
            gptable = self.gptable

        assert('Mloss' not in phewtable.columns), "'Mloss' field not found in phewtable."

        # Find all halos that ever hosted the gas particles in gptable
        halos = AccretionTracker.compile_halos_to_process(gptable)

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

        self.pptable = pptable
        return pptable

    @staticmethod
    def define_halo_relationship(progId, progHost, haloId, hostId):
        if(progId == 0): return "IGM"
        if(progId == haloId): return "SELF"
        if(progHost == haloId): return "PARENT"
        if(progId == hostId): return "SAT"
        if(progHost == hostId): return "SIB"
        return "IGM"

    @staticmethod
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
        halos['relation'] = halos.apply(
            lambda x : AccretionTracker.define_halo_relationship(
            x.progId, x.progHost, x.haloId, x.hostId), axis=1
        )

        return halos[['snapnum','haloId','relation']].set_index(['snapnum', 'haloId'])

    @staticmethod
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

        halos = AccretionTracker.compile_halos_to_process(gptable, ['snapnum','haloId'])
        halos = AccretionTracker.assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
        gptable = pd.merge(gptable, halos, how='left',
                           left_on=['snapnum','haloId'], right_index=True)
        gptable['relation'] = gptable['relation'].fillna('IGM')
        return gptable

    @staticmethod
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

        halos = AccretionTracker.compile_halos_to_process(pptable, ['snapfirst','birthId'])
        halos = AccretionTracker.assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
        pptable = pd.merge(pptable, halos, how='left',
                           left_on=['snapfirst','birthId'], right_index=True)
        pptable.rename(columns={'relation':'birthTag'}, inplace=True)
        return pptable

    @staticmethod
    def compile_halos_to_process(ptable, fields=['snapnum', 'haloId']):
        halos = ptable.loc[:, fields]
        halos.rename(columns={fields[0]:'snapnum', fields[1]:'haloId'}, inplace=True)
        halos = halos[halos.haloId != 0].drop_duplicates()
        return halos

    @staticmethod
    def compute_mgain_partition_by_pId(gptable, spark=None):
        '''
        For each gas particle in the gptable, compute its mass gain since last
        snapshot.
        '''
        if('Mgain' in gptable.columns):
            talk("Mgain is already found in gptable", "talky")
            return gptable
        if(spark is None):
            # This is a very expensive operation
            gptable = gptable.reset_index()
            gptable['Mgain'] = gptable.groupby(gptable.PId).Mass.diff().fillna(0.0)
            return gptable
        else:
            w = Window.partitionBy(gp.PId).orderBy(gp.snapnum)
            return gptable.withColumn('Mgain', gptable.Mass - sF.lag('Mass',1).over(w)).na.fill(0.0)


    def build_temporary_tables_for_galaxy(self, galIdTarget, spark=None):
        # Get the particle ID list to track
        pidlist = self._snap.get_gas_particles_in_galaxy(galIdTarget)        

        # Create/Load gptable
        self.build_gptable(pidlist, self.snapnum, 0)
        # Compute the 'Mgain' field for all gas particles in gptable
        self.gptable = self.__class__.compute_mgain_partition_by_pId(self.gptable)
        # Add their relations to galIdTarget
        self.gptable = self.__class__.add_relation_field_to_gptable(
            galIdTarget,
            self.gptable,
            self._snap._progtable,
            self._simulation._hostmap
        )

        # Create/Load pptable 
        self.build_pptable(
            self.gptable,
            self._simulation._inittable,
            self._simulation._phewtable
        )
        # Add birthtags relative to galIdTarget
        self.pptable = self.add_birthtag_field_to_pptable(
            galIdTarget,
            self.pptable,
            self._snap._progtable,
            self._simulation._hostmap
        )

    def verify_temporary_tables(self):
        '''
        Verify that self.gptables and self.pptables meets all requirements.
        '''

        ready = [False, False]
        if(self.gptable is None):
            print("gptable not loaded.")
        else if('Mgain' not in self.gptable.columns):
            print("gptable loaded but misses 'Mgain' field")
        else:
            print("gptable ready.")
            ready[0] = True

        if(self.pptable is None):
            print("pptable not loaded.")
        else if('Mloss' not in self.pptable.columns):
            print("pptable loaded but misses 'Mloss' field")            
        else if('birthTag' not in self.pptable.columns):
            print("pptable loaded but misses 'birthTag' field")
        else:
            print("pptable ready.")
            ready[1] = True
        return ready[0] & ready[1]

    def compute_wind_mass_partition_by_birthtag(self):
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

        if(not self.verify_temporary_tables()):
            print("gptable or pptable do not meet requirements yet.")
            return
        
        grps = self.pptable.groupby(['snapnum', 'haloId'])
        x = grps.apply(lambda x: x[['Mloss','birthTag']].groupby('birthTag').sum()).reset_index('birthTag')
        y = pd.merge(self.gptable, x, how='left', left_on=['snapnum', 'haloId'], right_on=['snapnum', 'haloId'])
        grps = y.groupby(['PId','snapnum'])
        mwindtable = grps.apply(lambda x : pd.DataFrame({
            'PId': x.PId,
            'snapnum': x.snapnum,
            'birthTag':x.birthTag,
            'Mgain':x.Mgain * x.Mloss / x.Mloss.sum()})
        )
        # mwindtable = z.groupby(['PId', 'birthTag']).sum()
        return mwindtable
        
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

import progen
from progen import *

if __mode__ == "__test__":
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

