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
__mode__ = "X"

import numpy as np
import pandas as pd
import snapshot
import utils
from utils import talk
from tqdm import tqdm

from config import cfg

import pyarrow as pa
import pyarrow.parquet as pq

import os
import pdb
import simulation

from importlib import reload

schema_gptable = {'columns':['PId','snapnum','Mass','haloId','Mgain','relation'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'Mass':'float64',
                            'haloId':'int32',
                            'Mgain':'float64',
                            'relation':'string'
                  }
}
schema_pptable = {'columns':['PId','haloId','snapnum','Mloss','snapfirst','birthId','birthTag'],
                  'dtypes':{'PId':'int64',
                            'snapnum':'int32',
                            'haloId':'int32',
                            'Mloss':'float64',
                            'snapfirst':'int32',                            
                            'birthId':'int32',
                            'birthTag':'string'
                  }
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
    
    def __init__(self, model, snapnum):
        self._snap = snapshot.Snapshot(model, snapnum)
        self._simulation = simulation.Simulation(model)
        self.model = model
        self.snapnum = snapnum
        self.path_base = self._simulation._path_tmpdir
        self.gptable = None
        self.pptable = None

    @classmethod
    def from_snapshot(cls, snap):
        act = cls(snap.model, snap.snapnum)
        act._snap = snap
        return act

    def initialize(self):
        '''
        Prepare all required permanent tables for accretion tracking. 
        Build the tables if they are not already existed.
        List of tables should be inplace:
        Simulation.inittable (inittable.csv)
        Simulation.phewtable, with Mloss (phewtable.parquet)
        Simulation.hostmap (hostmap.csv)
        Snapshot.progtable (progtab_{snapnum}.csv)
        '''
        
        talk("\nAccretionTracker: Initializing permanent tables...", "talky")
        self._simulation.build_inittable()
        try:
            self._simulation.load_phewtable()
        except:
            self._simulation.build_phewtable()
        # If the phewtable is half-built, then complete it and reload
        if('Mloss' not in self._simulation._phewtable.columns):
            self._simulation.compute_mloss_partition_by_pId(spark=spark)
            self._simulation.load_phewtable()            
        self._simulation.load_hostmap()
        self._snap.load_progtable()
        self._simulation.build_splittable()

    # def load_gptable(self, galIdTarget, verbose='talky'):
    #     fname = "gptable_{:03d}_{:05d}.parquet".format(self.snapnum, galIdTarget)
    #     path_gptable = os.path.join(self.path_base, fname)
    #     if(os.path.exists(path_gptable)):
    #         talk("Loading gptable from file.", 'quiet')
    #         gptable = pd.read_parquet(path_gptable)
    #         self.gptable = gptable
    #         return True
    #     else:
    #         talk("gptable {} not found. Use build_gptable() to build new table.".format(fname), verbose)
    #         return False

    # def build_gptable(self, pidlist, snapstart=0, include_stars=False, rebuild=False):
    #     '''
    #     Build the gptable for the particles to track. The particles are 
    #     specified by their particle IDs in the pidlist.
        
    #     Parameters
    #     ----------
    #     pidlist: list.
    #         List of particle IDs for which to build the table
    #     snapstart: int. Default = 0
    #         The first snapshot to start tracing the particles.
    #     include_stars: boolean. Default = False.
    #         If True, track the history star particles along with gas particles.
    #     rebuild: boolean. Default=False
    #         If True, rebuild the table even if a file exists.
    #     '''

    #     snaplast = self.snapnum
        
    #     fname = "gptable_{:03d}_{:05d}.parquet".format(self.snapnum, galIdTarget)
    #     path_gptable = os.path.join(self.path_base, fname)

    #     if(rebuild==False and os.path.exists(path_gptable)):
    #         talk("Load existing gptable.", 'normal')
    #         gptable = pd.read_parquet(path_gptable)
    #         self.gptable = gptable
    #         return gptable

    #     talk("Building gptable for {} gas particles".format(len(pidlist)), 'normal')
    #     gptable = None
    #     for snapnum in tqdm(range(snapstart, snaplast+1), desc='snapnum', ascii=True):
    #         snap = snapshot.Snapshot(self.model, snapnum, verbose='talky')
    #         snap.load_gas_particles(['PId','Mass','haloId'])

    #         # From pidlist infer all PIds needed from this snapshot
    #         # From the ancestor table, find the smallest snapnext that is larger than the current snapnum. For example, the history of particle Id=8 being:
    #         # PId   4   4   6   6   6   6   6   6   6   8   8   8
    #         # Gen   2   2   1   1   1   1   1   1   1   0   0   0
    #         # Mass  m/4 m/4 m/2 m/2 m/2 m/2 m/2 m/2 m/2 m   m   m
    #         #               s1              cur         s2
    #         # At current time, we found the next splitting at s2, where:
    #         # PId=8, parentId=6, snapnext=s2
    #         parents = self._ancestors.query('snapnext > @snapnum').reset_index()
    #         parents = parents.loc[parents.groupby('PId').gen.idxmax()].reset_index()
    #         pidtosearch = set(pidlist + list(parents.parentId.drop_duplicates()))
            
    #         gp = snap.gp.loc[snap.gp.PId.isin(pidtosearch), :]
    #         gp.loc[:,'snapnum'] = snapnum

    #         # Replace the ancestors' PIds at the current snapshot with the PIds of the descendents (when there are multiple descendents, make a copy for each descendent, INCLUDING itself (PId == parentId, but the particle has splitted)).
    #         # Might be multiple instances parentId -> multiple PId
    #         gp_parents = pd.merge(gp, parents[['PId', 'parentId', 'gen']],
    #                               left_on='PId', right_on='parentId',
    #                               suffixes=("_l", None))
    #         # Should reduce particle Mass according to its generation
    #         gp_parents.Mass = gp_parents.Mass / (2.0 ** gp_parents.gen)
    #         # gp_parents.drop(['PId_l', 'parentId', 'gen'], axis=1, inplace=True)
    #         gp_parents.drop(['PId_l', 'parentId'], axis=1, inplace=True)
    #         gp = gp[gp.PId.isin(pidlist)]
    #         gp = gp[~gp.PId.isin(gp_parents.PId)]

    #         if(include_stars):
    #             snap.load_star_particles(['PId','Mass','haloId'])
    #             # Early time, no star particle
    #             if(snap.sp is not None):
    #                 sp = snap.sp.loc[snap.sp.PId.isin(pidlist), :]
    #                 if(not sp.empty):
    #                     sp.loc[:,'snapnum'] = snapnum
    #                     gp = pd.concat([gp, sp])

    #         # The star particles and 'virgin' gas particle has gen=0.
    #         gp['gen'] = 0

    #         # Add those particles that have been a parent
    #         gp = pd.concat([gp, gp_parents])

    #         # Attach to the main table
    #         gptable = gp.copy() if gptable is None else pd.concat([gptable, gp])

    #     gptable = gptable.set_index('PId').sort_values('snapnum')
    #     self.gptable = gptable
    #     return gptable

    # def save_gptable(self, galIdTarget):
    #     fname = "gptable_{:03d}_{:05d}.parquet".format(self.snapnum, galIdTarget)
    #     path_gptable = os.path.join(self.path_base, fname)
    #     talk("Saving gptable as {}".format(path_gptable), 'quiet')

    #     schema = utils.pyarrow_read_schema(schema_gptable)
    #     tab = pa.Table.from_pandas(self.gptable, schema=schema, preserve_index=True)
    #     pq.write_table(tab, path_gptable)

    # def load_pptable(self, galIdTarget, verbose='talky'):
    #     fname = "pptable_{:03d}_{:05d}.parquet".format(self.snapnum, galIdTarget)
    #     path_pptable = os.path.join(self.path_base, fname)
    #     if(os.path.exists(path_pptable)):
    #         talk("Loading pptable from file.", 'quiet')
    #         pptable = pd.read_parquet(path_pptable)
    #         self.pptable = pptable
    #         return True
    #     else:
    #         talk("pptable {} not found. Use build_pptable() to build new table.".format(fname), verbose)
    #         return False

    # def build_pptable(self, inittable, phewtable, gptable=None, rebuild=False):
    #     '''
    #     Build or load a temporary table that stores all the necessary attributes 
    #     of selected PhEW particles that can be queried by the accretion tracking 
    #     engine.

    #     Parameters
    #     ----------
    #     inittable: pandas.DataFrame or Spark DataFrame
    #         Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
    #         The initial/final status of all PhEW particles
    #         Needed to know from which halo a PhEW particles were born
    #         Created from Simulation.build_inittable_from_simulation()
    #     phewtable: pandas.DataFrame or Spark DataFrame
    #         Columns: PId*, snapnum, Mass, haloId, (Mloss)
    #         All PhEW particles in any snapshot of the simulation
    #         The function return is a subset of it
    #         Created from Simulation.build_phewtable_from_simulation()
    #     gptable: pandas.DataFrame or Spark DataFrame. Default=None
    #         Columns: PId*, snapnum, Mass, haloId, ...
    #         Gas particles whose histories we are tracking.
    #         Needed to find the list of halos for processing
    #         If None, try self.gptable
    #     rebuild: boolean. Default=False
    #         If True, rebuild the table even if a file exists.

    #     Returns
    #     -------
    #     pptable: pandas.DataFrame or Spark DataFrame
    #         Columns: PId*, snapnum, haloId, Mloss, birthId
    #         Temporary table storing all the necessary attributes of selected PhEW
    #         particles that can be queried by the accretion tracking engine.
    #     '''
        
    #     path_pptable = os.path.join(self.path_base, "pptable.parquet")
    #     if(rebuild==False and os.path.exists(path_pptable)):
    #         talk("Load existing pptable.", 'normal')
    #         pptable = pd.read_parquet(path_pptable)
    #         self.pptable = pptable
    #         return pptable

    #     if(gptable is None):
    #         gptable = self.gptable

    #     assert('Mloss' in phewtable.columns), "'Mloss' field not found in phewtable."

    #     # Find all halos that ever hosted the gas particles in gptable
    #     halos = AccretionTracker.compile_halos_to_process(gptable)

    #     # Find all PhEW particles that ever appeared in these halos
    #     pptable = pd.merge(halos, phewtable, how='inner',
    #                        left_on=['snapnum', 'haloId'],
    #                        right_on=['snapnum', 'haloId'])
    #     # pptable: snapnum, haloId, PId, Mloss

    #     # Add the birth halo information to pptable
    #     pptable = pd.merge(pptable, inittable[['PId','snapfirst','birthId']],
    #                        how='left', left_on='PId', right_on='PId')

    #     self.pptable = pptable
    #     return pptable

    # def save_pptable(self, galIdTarget):
    #     fname = "pptable_{:03d}_{:05d}.parquet".format(self.snapnum, galIdTarget)
    #     path_pptable = os.path.join(self.path_base, fname)
    #     talk("Saving pptable as {}".format(path_pptable), 'quiet')

    #     schema = utils.pyarrow_read_schema(schema_pptable)
    #     tab = pa.Table.from_pandas(self.pptable, schema=schema, preserve_index=True)
    #     pq.write_table(tab, path_pptable)

    # @staticmethod
    # def _find_particle_ancestors(splittable, pidlist):
    #     '''
    #     In case there is splitting, we need to find all the ancestors of a 
    #     gas particle in order to figure out where its wind material came from 
    #     before it spawned from its parent.
    #     Furthermore, we also need to find all the split events for any of the 
    #     parents.

    #     Returns
    #     -------
    #     ancestors: pandas.DataFrame
    #         Columns: PId, parentId, snapnext, gen
    #     '''

    #     # Find the maximum generation of a particle when it was created/spawned
    #     maxgen = splittable.groupby('parentId')['parentGen'].max().to_dict()
        
    #     df = splittable.loc[splittable.PId.isin(pidlist),
    #                         ['PId','parentId','Mass','snapnext','parentGen']]

    #     # At the time when the PId was spawned
    #     df['gen'] = df['PId'].map(maxgen).fillna(0).astype('int32') + 1
    #     df_next = df.copy()
    #     i, max_iter = 0, 10
    #     # Find the parent of parent until no more
    #     while(i < max_iter):
    #         i = i + 1
    #         # Does the parent have a grandparent? (not self).
    #         # [<PId>, parentId*] -> [PId*, parentId]
    #         df_next = pd.merge(df_next[['PId', 'parentId', 'gen', 'parentGen']],
    #                            splittable,
    #                            how='inner', left_on='parentId', right_on='PId',
    #                            suffixes=("_l", None))
    #         # If no more ancestors found, break the loop
    #         if(df_next.empty):
    #             break
    #         # How many generations have passed between the time when the parent
    #         # was spawned and the time it spawns the PId?
    #         df_next['gen'] = df_next.gen \
    #             + df_next['parentId_l'].map(maxgen).fillna(0).astype('int32') \
    #             - df_next['parentGen_l'] + 1
    #         df_next.drop(['parentId_l','parentGen_l','atime','PId'], axis=1, inplace=True)
    #         df_next.rename(columns={'PId_l':'PId'}, inplace=True)
    #         df = pd.concat([df, df_next], axis=0)
    #     if(i == max_iter):
    #         warnings.warn("Maximum iterations {} reached.".format(max_iter))

    #     # Self-splitting
    #     df_self = splittable.loc[splittable.parentId.isin(pidlist+list(df.parentId)),
    #                              ['parentId','Mass','snapnext','parentGen']]
    #     # Does the parent have previous self-splitting?
    #     df_tmp = pd.merge(df[['PId','parentId','gen','parentGen']],
    #                       df_self[['parentId','parentGen','Mass','snapnext']],
    #                       how='inner', on='parentId',
    #                       suffixes=(None, "_r"))
    #     df_tmp = df_tmp.query('parentGen_r > parentGen')
    #     df_tmp['gen'] = df_tmp.gen \
    #         + df_tmp['parentGen_r'] - df_tmp['parentGen']
    #     df_tmp.drop(['parentGen_r'], axis=1, inplace=True)
            
    #     df_self['gen'] = df_self['parentGen']
    #     df_self['PId'] = df_self['parentId']

    #     # df_self contains particles out of the pidlist

    #     df = pd.concat([df, df_tmp, df_self], axis=0)
    #     # Sanity Check:
    #     # grp = df.groupby('PId')
    #     # grp.gen.count() - grp.gen.max() should be all 0
    #     return df[df.PId.isin(pidlist)]

    # Move to progen.py
    # @staticmethod
    # def define_halo_relationship(progId, progHost, haloId, hostId):
    #     if(progId == 0): return "IGM"
    #     if(progId == haloId): return "SELF"
    #     if(progHost == haloId): return "PARENT"
    #     if(progId == hostId): return "SAT"
    #     if(progHost == hostId): return "SIB"
    #     return "IGM"

    # Move to derivedtables
    # @staticmethod
    # def assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap):
    #     '''
    #     Map from the unique halo identifier (snapnum, haloId) to a descriptor for 
    #     its relationship with another halo (haloIdTarget) at a later time.
    #     In the accretion tracking engine, the table is built iteratively for each 
    #     halo of interest.

    #     Parameters
    #     ----------
    #     haloIdTarget: int.
    #         The haloId of the halo in the current snapshot.
    #     halos: pandas.DataFrame
    #         Columns: haloId*, snapnum*
    #         Halos uniquely identified with the (haloId, snapnum) pair
    #     progtable: pandas.DataFrame.
    #         Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
    #         Output of progen.find_all_previous_progenitors().
    #         Defines the progenitors of any halo in any previous snapshot.
    #     hostmap: pandas.DataFrame.
    #         Columns: snapnum*, haloId*, hostId
    #         Output of progen.build_haloId_hostId_map()
    #         Mapping between haloId and hostId for each snapshot.

    #     Returns:
    #     relation: pandas.DataFrame
    #         Columns: snapnum*, haloId*, relation
    #         Map between a halo (MultiIndex(snapnum, haloId)) to the relation, which 
    #         defines its relation to another halo (haloIdTarget) at a later time.
    #     '''

    #     progsTarget = progtable.loc[progtable.index == haloIdTarget,
    #                                 ['snapnum', 'progId', 'hostId']]
    #     progsTarget.rename(columns={'hostId':'progHost'}, inplace=True)

    #     halos = halos[['snapnum', 'haloId']].set_index(['snapnum', 'haloId'])

    #     # Find the hostId of each halo in its snapshot
    #     halos = halos.join(hostmap, how='left') # (snapnum, haloId) -> hostId
    #     halos = halos.reset_index()

    #     halos = pd.merge(halos, progsTarget, how='left',
    #                      left_on = 'snapnum', right_on = 'snapnum')
    #     halos['relation'] = halos.apply(
    #         lambda x : AccretionTracker.define_halo_relationship(
    #         x.progId, x.progHost, x.haloId, x.hostId), axis=1
    #     )

    #     return halos[['snapnum','haloId','relation']].set_index(['snapnum', 'haloId'])

    # @staticmethod
    # def add_relation_field_to_gptable(haloIdTarget, gptable, progtable, hostmap):
    #     '''
    #     Add a field ('relation') in the gptable that defines the relation between 
    #     haloIdTarget and any halo that hosts a gas particle in the gptable.

    #     Parameters
    #     ----------
    #     haloIdTarget: int.
    #         The haloId of the halo in the current snapshot.
    #     gptable: pandas.DataFrame or Spark DataFrame
    #     progtable: pandas.DataFrame.
    #         Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
    #         Output of progen.find_all_previous_progenitors().
    #         Defines the progenitors of any halo in any previous snapshot.
    #     hostmap: pandas.DataFrame.
    #         Columns: snapnum*, haloId*, hostId
    #         Output of progen.build_haloId_hostId_map()
    #         Mapping between haloId and hostId for each snapshot.

    #     Returns:
    #     gptable: pandas.DataFrame
    #     '''

    #     halos = AccretionTracker.compile_halos_to_process(gptable, ['snapnum','haloId'])
    #     halos = AccretionTracker.assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
    #     gptable = pd.merge(gptable, halos, how='left',
    #                        left_on=['snapnum','haloId'], right_index=True)
    #     gptable['relation'] = gptable['relation'].fillna('IGM')
    #     return gptable

    # @staticmethod
    # def add_birthtag_field_to_pptable(haloIdTarget, pptable, progtable, hostmap):
    #     '''
    #     Add a field ('birthTag') in the pptable that defines the relation between 
    #     haloIdTarget and the halo where a PhEW particle was born.

    #     Parameters
    #     ----------
    #     haloIdTarget: int.
    #         The haloId of the halo in the current snapshot.
    #     pptable: pandas.DataFrame or Spark DataFrame
    #         Columns: PId*, snapnum, Mass, haloId, snapfirst, birthId
    #         PhEW particles that shared mass with the gas particles
    #     progtable: pandas.DataFrame.
    #         Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
    #         Output of progen.find_all_previous_progenitors().
    #         Defines the progenitors of any halo in any previous snapshot.
    #     hostmap: pandas.DataFrame.
    #         Columns: snapnum*, haloId*, hostId
    #         Output of progen.build_haloId_hostId_map()
    #         Mapping between haloId and hostId for each snapshot.

    #     Returns:
    #     pptable: pandas.DataFrame
    #     '''

    #     halos = AccretionTracker.compile_halos_to_process(pptable, ['snapfirst','birthId'])
    #     halos = AccretionTracker.assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
    #     pptable = pd.merge(pptable, halos, how='left',
    #                        left_on=['snapfirst','birthId'], right_index=True)
    #     pptable.rename(columns={'relation':'birthTag'}, inplace=True)
    #     return pptable

    # Move to DerivedTable
    # @staticmethod
    # def compile_halos_to_process(ptable, fields=['snapnum', 'haloId']):
    #     halos = ptable.loc[:, fields]
    #     halos.rename(columns={fields[0]:'snapnum', fields[1]:'haloId'}, inplace=True)
    #     halos = halos[halos.haloId != 0].drop_duplicates()
    #     return halos

    # Move to GasPartTable
    # @staticmethod
    # def compute_mgain_partition_by_pId(gptable, spark=None):
    #     '''
    #     For each gas particle in the gptable, compute its mass gain since last
    #     snapshot. This does not account for splitting particles yet. 
    #     '''
    #     if('Mgain' in gptable.columns):
    #         talk("Mgain is already found in gptable", "talky")
    #         return gptable
    #     if(spark is None):
    #         # This is a very expensive operation
    #         gptable = gptable.reset_index()
    #         gptable['Mgain'] = gptable.groupby(gptable.PId).Mass.diff().fillna(0.0)
    #         return gptable
    #     else:
    #         w = Window.partitionBy(gp.PId).orderBy(gp.snapnum)
    #         return gptable.withColumn('Mgain', gptable.Mass - sF.lag('Mass',1).over(w)).na.fill(0.0)

    @staticmethod
    def update_mgain_for_split_events(gptable, splittable):
        '''
        In case there was splitting between the last snapshot and the current 
        snapshot, gather information from the splittable and update the rows 
        where a split happened.
          - If this is a new born particle, 
            define mgain = gptable.Mass - splittable.Mass / 2
          - If this is a splitting particle, 
            define mgain = mgain + splittable.Mass / 2
        '''

        # Splitted particle, add back the lost portion
        df = pd.merge(gptable, ancestors, how='left',
                      left_on=['PId','snapnum'], right_on=['PId','snapnext'],
                      suffixes=(None, '_r'))
        df.Mass_r = df.Mass_r.fillna(0.0) / 2.0
        df.Mgain = df.Mgain + df.Mass_r

        
        df_spt = [['parentId','snapnext','Mass']]
        df_spt['dMass'] = df_spt['Mass'] / 2
        df_spt.rename(columns={'parentId':'PId'}, inplace=True)

        df = pd.concat([df_new['PId','snapnext','dMass'],
                        df_spt['PId','snapnext','dMass']], axis=0)

        df_spt = df_spt.groupby(['PId','snapnext'])['dMass'].sum()
        df_spt = pd.merge(gptable, df_spt, how='inner',
                          left_on=['PId','snapnum'], right_index=True)
        df_spt['dMass'] = df_spt['dMass'].fillna(0.0)
        df_spt['Mgain'] = df_spt['Mgain'] + df_spt['dMass']

        df_new = pd.merge(gptable, df_new[['PId','snapnext','dMass']], how='inner',
                          left_on=['PId','snapnum'], right_on=['PId','snapnext'])
        df_new['dMass'] = df_new['dMass'].fillna(0.0)
        df_new['Mgain'] = df_new['Mass'] - df_new['dMass']

        df = pd.concat([df_new, df_spt], axis=0)
        
        # Now select only parentIds that are in the gptable
        df.drop(['dMass'], axis=1, inplace=True)
        return df

    def build_temporary_tables_for_galaxy(self, galIdTarget, rebuild=False, include_stars=False, spark=None):
        '''
        Build gptable and pptable for particles selected from galIdTarget.
        The permanent tables, inittable, phewtable, hostmap, progtable should 
        already be in place with the initialize() call.

        Parameters
        ----------
        galIdTarget: int.
            The galId of the target galaxy.
        include_stars: boolean. Default = False.
            If True, track the history star particles along with gas particles.
        spark: 
        rebuild: boolean. Default = False.
            If True, rebuild the gptable, otherwise load existing file.
        '''
        
        talk("\nAccretionTracker: Building temporary tables for galId={} at snapnum={}".format(galIdTarget, self.snapnum), "talky")
        # Get the particle ID list to track
        if(not self.load_gptable(galIdTarget, verbose='quiet') or rebuild==True):
            pidlist = self._snap.get_gas_particles_in_galaxy(galIdTarget)
            if(include_stars):
                pidlist += self._snap.get_star_particles_in_galaxy(galIdTarget)

            # In case of splitting, find all ancestors of any gas particle
            self._ancestors = self.__class__._find_particle_ancestors(self._simulation._splittable, pidlist)

            # Create/Load gptable
            self.build_gptable(pidlist, include_stars=include_stars, rebuild=True)
            # Compute the 'Mgain' field for all gas particles in gptable
            self.gptable = self.__class__.compute_mgain_partition_by_pId(self.gptable)
            # Add their relations to galIdTarget
            self.gptable = self.__class__.add_relation_field_to_gptable(
                galIdTarget,
                self.gptable,
                self._snap._progtable,
                self._simulation._hostmap
            )
            self.save_gptable(galIdTarget)

        # Create/Load pptable
        if(not self.load_pptable(galIdTarget, verbose='quiet') or rebuild == True):
            self.build_pptable(
                self._simulation._inittable,
                self._simulation._phewtable,
                rebuild=True
            )
            # Add birthtags relative to galIdTarget
            self.pptable = self.add_birthtag_field_to_pptable(
                galIdTarget,
                self.pptable,
                self._snap._progtable,
                self._simulation._hostmap
            )
            self.save_pptable(galIdTarget)

    def verify_temporary_tables(self):
        '''
        Verify that self.gptables and self.pptables meets all requirements.
        '''

        ready = [False, False]
        if(self.gptable is None):
            print("gptable not loaded.")
        elif('Mgain' not in self.gptable.columns):
            print("gptable loaded but misses 'Mgain' field")
        else:
            print("gptable ready.")
            ready[0] = True

        if(self.pptable is None):
            print("pptable not loaded.")
        elif('Mloss' not in self.pptable.columns):
            print("pptable loaded but misses 'Mloss' field")            
        elif('birthTag' not in self.pptable.columns):
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

        # Do for each halo (snapnum, haloId)
        grps = self.pptable.groupby(['snapnum', 'haloId'])
        # total mloss to the halo by birthTag
        x = grps.apply(lambda x: x[['Mloss','birthTag']].groupby('birthTag').sum()).reset_index('birthTag')
        y = pd.merge(self.gptable, x, how='left', left_on=['snapnum', 'haloId'], right_on=['snapnum', 'haloId'])
        # Mloss = NaN. no corresponding PhEW in that halo
        y.birthTag = y.birthTag.fillna('IGM')
        y.Mloss = y.Mloss.fillna(0.0)
        grps = y.groupby(['PId','snapnum'])
        # each grp is a particle at a snapnum
        # if not found, birthTag = NaN, but Mgain should be IGM?
        mwtable = grps.apply(lambda x : pd.DataFrame({
            'PId': x.PId,
            'snapnum': x.snapnum,
            'birthTag':x.birthTag,
            'Mgain':x.Mgain * \
            (1.0 if x.Mloss.sum() == 0.0 else x.Mloss / x.Mloss.sum())})
        )
        # mwindtable = z.groupby(['PId', 'birthTag']).sum()
        return mwtable
        
    def gp_mass_gain_since_last_snapshot(PId, snapnum, gptable):
        '''
        Find the amount of mass gained in a normal gas particle (PId) between the 
        previous snapshot (snapnum-1) and the current snapshot (snapnum)
        '''

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

def get_ism_history_for_galaxy(galIdTarget):
    # Look at the current ISM particles of a galaxy
    model = "l25n144-test"    
    snap = snapshot.Snapshot(model, 108)
    act = AccretionTracker.from_snapshot(snap)
    act.initialize()
    act.build_temporary_tables_for_galaxy(galIdTarget)
    mwtable = act.compute_wind_mass_partition_by_birthtag()
    return mwtable, act

galIdTarget = 715 # Npart = 28, logMgal = 9.5, within 0.8-0.9 range

# 568, 715, 1185

__mode__ = "__showX__"

if(__mode__ == "__load__"):
    model = "l25n144-test"    
    snap = snapshot.Snapshot(model, 108)
    act = AccretionTracker.from_snapshot(snap)
    act.initialize()
    act.build_temporary_tables_for_galaxy(galIdTarget, rebuild=False)
    mwtable = act.compute_wind_mass_partition_by_birthtag()
    mwtable.groupby(['PId','birthTag'])['Mgain'].sum()

