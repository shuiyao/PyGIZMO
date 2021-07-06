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

import pyarrow as pa
import pyarrow.parquet as pq

import os
import pdb
import simulation

from importlib import reload

from derivedtables import GasPartTable, PhEWPartTable

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

# Parquet I/O
#https://blog.datasyndrome.com/python-and-parquet-performance-e71da65269ce

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

