'''
Procedures at the simulation level.
'''

import os
import glob
from bisect import bisect_right

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

from . import winds
from .utils import *
from .config import SimConfig
from . import snapshot
from . import progen

# from sparkutils import *

class Simulation(object):
    '''
    Top level APIs for simulation.
    
    Example
    -------
    >>> model = "l25n144-test"
    >>> sim = Simulation(model)
    Simulation: l25n144-test
    >>> print(f"Number of snapshots: {sim.nsnaps}")
    Number of snapshots: 109
    >>> sim.load_timeinfo_for_snapshots()
               a      zred
    0    0.03226  29.99997
    1    0.04762  20.00002
    2    0.06250  15.00000
    3    0.06667  13.99999
    4    0.07143  12.99999
    ..       ...       ...
    104  0.90909   0.10000
    105  0.93023   0.07500
    106  0.95238   0.05000
    107  0.97561   0.02500
    108  1.00000   0.00000
    <BLANKLINE>
    [109 rows x 2 columns]
    '''
    
    def __init__(self, model, config=SimConfig(), verbose=None):
        '''
        Parameters
        ----------
        model: String. 
            The name of the simulation.
        config: SimConfig. Default=SimConfig()
            The configuration file to load. By default, load the pygizmo.cfg
        '''
        
        self._model = model
        self._cfg = config
        self._path_data = os.path.join(self._cfg.get('Paths', 'data'), model)
        self._path_workdir = os.path.join(self._cfg.get('Paths', 'workdir'), model)
        self._path_tmpdir = os.path.join(self._cfg.get('Paths', 'tmpdir'), model)
        self._path_winds = os.path.join(self._path_data, "WINDS")
        if(not os.path.exists(self._path_workdir)):
            os.mkdir(self._path_workdir)
        self._n_snaps = None

        self.verbose = verbose

        talk(self.__str__(), "normal", self.verbose)        

    def __str__(self):
        return f"Simulation: {self._model}"

    def __repr__(self):
        return f"Simulation({self._model!r})"

    @staticmethod
    def load_timeinfo_for_snapshots(fredz="data/redshift.txt"):
        '''
        Load the correspondence between snapnum and redshift and cosmic time.
        '''
        fredz = abspath(fredz)
        df_redz = pd.read_csv(fredz, sep='\s+', header=0)
        df_redz = df_redz.drop("snapnum", axis=1)
        return df_redz

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

def _test(model=None):
    model = "l25n144-test" if (model is None) else model
    sim = Simulation(model)
    print(f"Number of snapshots: {sim.nsnaps}")
    print(sim.load_timeinfo_for_snapshots())

if(__name__ == "__main__"):
    _test()

