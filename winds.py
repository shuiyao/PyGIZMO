'''
Procedures for processing the PhEW (wind) particles.
'''

import os
import glob
import warnings

import pandas as pd

from . import utils

#from sparkutils import *

DefaultSchema = utils.load_default_schema()

def read_split(path_winds, nfiles=0, sep=',', schema=DefaultSchema):
    if(nfiles):
        n = nfiles
    else:
        n = len(glob.glob(os.path.join(path_winds, "split.*")))
    assert (n > 0), "Can not find any split.* under {}".format(path_winds)
    print("Reading split from {} files.".format(n))
    cols = schema['split']['columns'].copy()
    for i in range(n):
        filename = os.path.join(path_winds, "split.{}".format(i))
        skip = 1 if i==0 else 0
        dfnew = pd.read_csv(filename, sep=sep, skiprows=skip,
                            names=cols,
                            dtype=schema['split']['dtypes'])
        df = dfnew.copy() if i == 0 else pd.concat([df, dfnew])
    return df

def read_initwinds(path_winds, nfiles=0, sep=',', columns=None, schema=DefaultSchema, minPotIdField=True):
    if(nfiles):
        n = nfiles
    else:
        n = len(glob.glob(os.path.join(path_winds, "initwinds.*")))
    assert (n > 0), "Can not find any initwinds.* under {}".format(path_winds)
    print("Reading initwinds from {} files.".format(n))
    cols = schema['initwinds']['columns'].copy()
    if(not minPotIdField):
        cols.remove('MinPotID')
    for i in range(n):
        filename = os.path.join(path_winds, "initwinds.{}".format(i))
        skip = 1 if i==0 else 0
        dfnew = pd.read_csv(filename, sep=sep, skiprows=skip,
                            names=cols,
                            dtype=schema['initwinds']['dtypes'])
        if(columns is not None):
            dfnew = dfnew.loc[:,columns]
        df = dfnew.copy() if i == 0 else pd.concat([df, dfnew])
        # Note 1. See end
    return df

def read_rejoin(path_winds, nfiles=0, sep=',', columns=None, schema=DefaultSchema):
    if(nfiles):
        n = nfiles
    else:
        n = len(glob.glob(os.path.join(path_winds, "rejoin.*")))
    assert (n > 0), "Can not find any rejoin.* under {}".format(path_winds)
    print("Reading rejoin info from {} files.".format(n))
    for i in range(n):
        filename = os.path.join(path_winds, "rejoin.{}".format(i))
        skip = 1 if i==0 else 0
        dfnew = pd.read_csv(filename, sep=sep, skiprows=skip,
                            names=schema['rejoin']['columns'],
                            dtype=schema['rejoin']['dtypes'])
        if(columns is not None):
            dfnew = dfnew.loc[:,columns]
        df = dfnew.copy() if i == 0 else pd.concat([df, dfnew])
        # Note 1. See end
    return df

def read_phews(path_winds, nfiles=0, columns=None, schema=DefaultSchema):
    if(nfiles):
        n = nfiles
    else:
        n = len(glob.glob(path_winds+"phews.*"))
    assert (n > 0), "Can not find any phews.* under {}".format(path_winds)
    print("Reading phews from {} files.".format(n))
    for i in range(n):
        filename = os.path.join(path_winds, "phews.{}".format(i))
        skip = 1 if i==0 else 0
        dfnew = pd.read_csv(filename, sep='\s+', skiprows=skip,
                            names=schema['phews']['columns'],
                            dtype=schema['phews']['dtypes'])
        if(columns is not None):
            dfnew = dfnew[columns]
        df = dfnew.copy() if i == 0 else pd.concat([df, dfnew])
        # Note 1. See end
    return df

def spark_read_initwinds(path_winds, columns=None, schema=DefaultSchema):
    schemaSpark = spark_read_schema(schema['initwinds'])
    path_files = os.path.join(path_winds, "initwinds.*")
    sdf = spark.read.options(delimiter=' ').csv(path_files, schemaSpark)
    if(columns is not None):
        sdf = sdf.select(columns)
    return sdf

def spark_read_rejoin(path_winds, columns=None, schema=DefaultSchema):
    schemaSpark = spark_read_schema(schema['rejoin'])
    path_files = os.path.join(path_winds, "rejoin*")
    sdf = spark.read.options(delimiter=' ').csv(path_files, schemaSpark)
    if(columns is not None):
        sdf = sdf.select(columns)
    return sdf


__mode__ = "__x__"
if __mode__ == "__test__":
    from myinit import *
    import time
    model = "l50n288-phew-m5"
    AMAX = 0.505
    snapstr = "078"
    path_model = os.path.join(DIRS['DATA'], model)
    path_winds = os.path.join(path_model, "WINDS/")
    tbeg = time.time()
    # df = read_initwinds(path_winds, columns=['atime', 'PhEWKey', 'Mass', 'PID'])
    df = read_phews(path_winds, columns=['atime', 'PhEWKey', 'dr', 'dv', 'M_cloud'])
    print("Cost: {} s".format(time.time() - tbeg))


# Note 1:
# Alternatively, one could define an empty dataframe df first. But it takes twice as much time as the current version. In addition, if dtypes was not specified in creating the empty dataframe, the end result will not find correct dtypes for PhEWKey and PID.
