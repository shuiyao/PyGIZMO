'''
Procedures at the simulation level.
'''

from myinit import *
import utils
from utils import talk
import pandas as pd
import numpy as np
from config import cfg
import glob
import winds
import os
from bisect import bisect_right
from tqdm import tqdm
import snapshot

PATHS = cfg['Paths']

model = "l25n144-phew-rcloud"
class Simulation():
    def __init__(self, model):
        self._model = model
        self._path_data = os.path.join(PATHS['data'], model)
        self._path_workdir = os.path.join(PATHS['workdir'], model)
        self._path_tmpdir = os.path.join(PATHS['tmp'], model)
        self._path_winds = os.path.join(self._path_data, "WINDS")
        if(not os.path.exists(self._path_workdir)):
            os.mkdir(self._path_workdir)
        self._n_snaps = None

    def build_inittable_from_simulation(self, overwrite=False):
        '''
        Gather all PhEW particles within the entire simulation and find their 
        initial and final attributes. Returns an inittable that will be queried
        extensively by the wind tracking engine.

        Parameters
        ----------
        overwrite: boolean. Default=False.
            If True, force creating the table.
        
        Returns
        -------
        inittable: pandas.DataFrame.
            The initial and final attributes of all PhEW particles
            Columns: PId*, snapfirst, minit, snaplast, mlast
            snapfirst is the snapshot BEFORE it was launched as a wind
            snaplast is the snapshot AFTER it stopped being a PhEW
        '''
        schema = {'columns':['PId','snapfirst','minit','snaplast','mlast'],
                  'dtypes':{'PId':'int64',
                            'snapfirst':'int32',
                            'minit':'float32',
                            'snaplast':'int32',
                            'mlast':'float32'}
        }
        fout = os.path.join(self._path_workdir, "inittable.csv",)

        # Load if existed.
        if(os.path.exists(fout) and overwrite == False):
            talk("Loading existing inittable.csv file...", 'normal')
            return pd.read_csv(fout, dtype=schema['dtypes'])

        # Create new if not existed.
        dfi = winds.read_initwinds(path_winds, columns=['atime','PhEWKey','Mass','PID'], minPotIdField=False)
        dfr = winds.read_rejoin(path_winds, columns=['atime','PhEWKey','Mass'])

        redz = utils.load_timeinfo_for_snapshots()
        dfi['snapfirst'] = dfi.atime.map(lambda x : bisect_right(redz.a, x)-1)
        dfi.rename(columns={'Mass':'minit','PID':'PId'}, inplace=True)
        dfr['snaplast'] = dfr.atime.map(lambda x : bisect_right(redz.a, x))
        dfr.rename(columns={'Mass':'mlast'}, inplace=True)

        df = pd.merge(dfi, dfr, how='left', left_on='PhEWKey', right_on='PhEWKey')
        df.snaplast = df.snaplast.fillna(109).astype('int32')
        df.mlast = df.mlast.fillna(0.0).astype('float32')
        df = df[['PId','snapfirst','minit','snaplast','mlast']]
        df.to_csv(fout, index=False)
        return df

    def build_phewtable_from_simulation(self, snaplast=None, snapstart=0):
        '''
        Build a gigantic table that contains all PhEW particles ever appeared in a
        simulation. The initial status and the final status of a PhEW particle is
        found in initwinds.* and rejoin.* files. Any one record corresponds to a 
        PhEW particle at a certain snapshot.
        Opt for parallel processing.
        '''

        if(snaplast is None):
            snaplast = self.nsnaps
            
        phewtable = None
        for snapnum in tqdm(range(snapstart, snaplast+1), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(sim.model, snapnum)
            snap.load_gas_particles(['PId','Mass','Mc'])
            gp = snap.gp.loc[snap.gp.Mc > 0, ['PId','Mass']]    
            gp.loc[:,'snapnum'] = snapnum
            phewtable = gp.copy() if phewtable is None else pd.concat([phewtable, gp])

        inittable = self.build_inittable_from_simulation()
        tmp = inittable[['PId','minit','snapfirst']]\
            .rename(columns={'minit':'Mass','snapfirst':'snapnum'})
        phewtable = pd.concat([phewtable, tmp])
        tmp = inittable[['PId','mlast','snaplast']]\
            .rename(columns={'mlast':'Mass','snaplast':'snapnum'})
        phewtable = pd.concat([phewtable, tmp])

        # Very computationally intensive
        phewtable = phewtable.set_index('PId').sort_values('snapnum')
        return phewtable

    @property
    def model(self):
        return self._model
    
    @property
    def nsnaps(self):
        '''
        Total number of snapshots in the simulation.
        '''
        if(self._n_snaps is None):
            files = glob.glob(os.path.join(self._path_data, "snapshot_*.hdf5"))
            self._n_snaps = len(files)
            return self._n_snaps
        else:
            return self._n_snaps

        

sim = Simulation(model)
df = sim.build_phewtable_from_simulation(snapstart=100, snaplast=108)








