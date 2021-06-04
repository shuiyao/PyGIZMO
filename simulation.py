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
import pyarrow as pa
import pyarrow.parquet as pq

from sparkutils import *

PATHS = cfg['Paths']

schema_phewtable = {'columns':['PId','snapnum','Mass','haloId'],
                    'dtypes':{'PId':'int64',
                              'snapnum':'int32',
                              'Mass':'float64',
                              'haloId':'int32'}
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


# model = "l12n144-phew-movie-200"
model = "l25n144-test"
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

    def build_inittable_from_simulation(self, overwrite=False, spark=None):
        '''
        Gather all PhEW particles within the entire simulation and find their 
        initial and final attributes. Returns an inittable that will be queried
        extensively by the wind tracking engine.

        Parameters
        ----------
        overwrite: boolean. Default=False.
            If True, force creating the table.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.
        
        Returns
        -------
        inittable: pandas.DataFrame or Spark DataFrame
            The initial and final attributes of all PhEW particles
            Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
            snapfirst is the snapshot BEFORE it was launched as a wind
            snaplast is the snapshot AFTER it stopped being a PhEW
        '''
        fout = os.path.join(self._path_workdir, "inittable.csv",)

        # Load if existed.
        if(os.path.exists(fout) and overwrite == False):
            talk("Loading existing inittable.csv file...", 'normal')
            if(spark is not None):
                schemaSpark = spark_read_schema(schema_inittable)
                return spark.read.option('header','true').csv(fout, schemaSpark)
            else:
                return pd.read_csv(fout, dtype=schema_inittable['dtypes'])

        # Create new if not existed.
        dfi = winds.read_initwinds(self._path_winds, columns=['atime','PhEWKey','Mass','PID'], minPotIdField=True)
        redz = utils.load_timeinfo_for_snapshots()
        dfi['snapfirst'] = dfi.atime.map(lambda x : bisect_right(redz.a, x)-1)
        dfi.rename(columns={'Mass':'minit','PID':'PId'}, inplace=True)
        grps = dfi.groupby('snapfirst')
        
        talk("Looking for the halos where PhEW particles are born.", "normal")
        frames = []
        for snapnum in tqdm(range(0, sim.nsnaps), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(sim.model, snapnum)
            snap.load_gas_particles(['PId','haloId'])
            df = pd.merge(grps.get_group(snapnum), snap.gp, how='left',
                          left_on='PId', right_on='PId')
            frames.append(df)
        dfi = pd.concat(frames, axis=0).rename(columns={'haloId':'birthId'})
        dfi.birthId = dfi.birthId.fillna(0).astype('int32')

        # Load final status of the winds and merge
        dfr = winds.read_rejoin(self._path_winds, columns=['atime','PhEWKey','Mass'])
        dfr['snaplast'] = dfr.atime.map(lambda x : bisect_right(redz.a, x))
        dfr.rename(columns={'Mass':'mlast'}, inplace=True)
        df = pd.merge(dfi, dfr, how='left', left_on='PhEWKey', right_on='PhEWKey')
        df.snaplast = df.snaplast.fillna(109).astype('int32')
        df.mlast = df.mlast.fillna(0.0).astype('float32')

        df = df[['PId','snapfirst','minit','birthId','snaplast','mlast']]

        df.to_csv(fout, index=False)
        return df

    def build_phewtable_from_simulation(self, snaplast=None, snapstart=0, overwrite=False, ignore_init=False, spark=None):
        '''
        Build a gigantic table that contains all PhEW particles ever appeared in a
        simulation. The initial status and the final status of a PhEW particle is
        found in initwinds.* and rejoin.* files. Any one record corresponds to a 
        PhEW particle at a certain snapshot.
        Opt for parallel processing.

        Parameters
        ----------
        snapstart: int. Default=0
        snaplast: int. Default=None
            If None. Use all snapshot until the end of the simulation.
        overwrite: boolean. Default=False
            If True, force creating the table and overwrite existing file.
        ignore_init: boolean. Default=False
            If True, do not include the initial/final state of the particles
            in the phewtable.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.

        Returns
        -------
        phewtable: pandas.DataFrame or Spark DataFrame
            A gigantic table containing all PhEW particles in any snapshot.
            Columns: PId*, snapnum, Mass, haloId
            This table will be heavily queried by the accretion tracking engine.

        OutputFiles
        -----------
        data/phewtable.parquet
        '''

        path_phewtable = os.path.join(self._path_data, "phewtable.parquet")
        if(overwrite==False and os.path.exists(path_phewtable)):
            talk("Load existing phewtable.", 'talky')
            if(spark is not None):
                schemaSpark = spark_read_schema(schema_phewtable)
                return spark.read.parquet(path_phewtable)
            else:
                phewtable = pd.read_parquet(path_phewtable)
                return phewtable
        
        if(snaplast is None):
            snaplast = self.nsnaps
            
        phewtable = None
        for snapnum in tqdm(range(snapstart, snaplast+1), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(sim.model, snapnum)
            snap.load_gas_particles(['PId','Mass','Mc','haloId'])
            gp = snap.gp.loc[snap.gp.Mc > 0, ['PId','Mass','haloId']]    
            gp.loc[:,'snapnum'] = snapnum
            phewtable = gp.copy() if phewtable is None else pd.concat([phewtable, gp])

        if (not ignore_init):
            inittable = self.build_inittable_from_simulation()
            tmp = inittable[['PId','minit','snapfirst']]\
                .rename(columns={'minit':'Mass','snapfirst':'snapnum'})
            tmp['haloId'] = 0 # Unknown
            phewtable = pd.concat([phewtable, tmp])
            tmp = inittable[['PId','mlast','snaplast']]\
                .rename(columns={'mlast':'Mass','snaplast':'snapnum'})
            tmp['haloId'] = 0 # Unknown        
            phewtable = pd.concat([phewtable, tmp])

        # Very computationally intensive
        phewtable = phewtable.sort_values(['PId','snapnum'])

        # This will take for-ever
        # phewtable['dM'] = phewtable.groupby('PId')['Mass'].diff().fillna(0.0)

        # Write parquet file.
        schema = utils.get_pyarrow_schema_from_json(schema_phewtable)
        tab = pa.Table.from_pandas(phewtable, schema=schema, preserve_index=False)
        pq.write_table(tab, path_phewtable)
        
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

def compute_mloss_partition_by_pId(phewtable, spark=None):
    '''
    For each PhEW particle in the phewtable, compute its mass loss since last
    snapshot (it could have just launched after the last snapshot).
    '''
    if(spark is None):
        # This is a very expensive operation
        phewtable['Mloss'] = phewtable.groupby('PId').diff().fillna(0.0)
        return phewtable
    else:
        w = Window.partitionBy(phewtable.PId).orderBy(phewtable.snapnum)
        return sdf.withColumn('Mloss', sdf.Mass - sF.lag('Mass',1).over(w))\
                  .na.fill(0.0)

sim = Simulation(model)
df = sim.build_inittable_from_simulation(overwrite=False)
# df = sim.build_phewtable_from_simulation(snapstart=100, snaplast=108, ignore_init=True)




