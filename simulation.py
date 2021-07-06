'''
Procedures at the simulation level.
'''

import utils
from utils import talk
import pandas as pd
import numpy as np
from config import SimConfig
import glob
import winds
import os
from bisect import bisect_right
from tqdm import tqdm
import snapshot
import progen
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

# from sparkutils import *

schema_phewtable = {'columns':['PId','snapnum','Mass','haloId','Mloss'],
                    'dtypes':{'PId':'int64',
                              'snapnum':'int32',
                              'Mass':'float64',
                              'haloId':'int32',
                              'Mloss':'float64'
                    }
}
schema_inittable = {'columns':['PId','snapfirst','minit','birthId',
                               'snaplast','mlast'],
                    'dtypes':{'PId':'int64',
                              'snapfirst':'int32',
                              'minit':'float64',
                              'birthId':'int32',
                              'snaplast':'int32',
                              'mlast':'float64'}
}
schema_splittable = {'columns':['PId','parentId','Mass','atime','snapnext','parentGen'],
                    'dtypes':{'PId':'int64',
                              'parentId':'int64',
                              'Mass':'float64',
                              'atime':'float32',
                              'snapnext':'int32',
                              'parentGen':'int32'}
}

model = "l25n144-test"

class Simulation():
    def __init__(self, model, config=SimConfig()):
        self._model = model
        self._cfg = config
        self._path_data = os.path.join(self._cfg.get('Paths', 'data'), model)
        self._path_workdir = os.path.join(self._cfg.get('Paths', 'workdir'), model)
        self._path_tmpdir = os.path.join(self._cfg.get('Paths', 'tmpdir'), model)
        self._path_winds = os.path.join(self._path_data, "WINDS")
        if(not os.path.exists(self._path_workdir)):
            os.mkdir(self._path_workdir)
        self._n_snaps = None

        self._inittable = None
        self._phewtable = None
        self._hostmap = None
        self._splittable = None        

    def __str__(self):
        return f"Simulation: {self._model}"

    def __repr__(self):
        return f"Simulation({self._model!r})"

    @staticmethod
    def load_timeinfo_for_snapshots(fredz="data/redshift.txt"):
        '''
        Load the correspondence between snapnum and redshift and cosmic time.
        '''
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



