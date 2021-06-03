'''
Procedures for processing the PhEW (wind) particles.
'''

import pandas as pd
import warnings
import os
import glob
import utils
import pdb

DefaultSchema = utils.load_default_schema()

def read_initwinds(path_winds, nfiles=0, columns=None, schema=DefaultSchema, minPotIdField=True):
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
        dfnew = pd.read_csv(filename, sep='\s+', skiprows=skip,
                            names=cols,
                            dtype=schema['initwinds']['dtypes'])

        if(columns is not None):
            dfnew = dfnew.loc[:,columns]
        df = dfnew.copy() if i == 0 else pd.concat([df, dfnew])
        # Note 1. See end
    return df

def read_rejoin(path_winds, nfiles=0, columns=None, schema=DefaultSchema):
    if(nfiles):
        n = nfiles
    else:
        n = len(glob.glob(os.path.join(path_winds, "rejoin.*")))
    assert (n > 0), "Can not find any rejoin.* under {}".format(path_winds)
    print("Reading rejoin info from {} files.".format(n))
    for i in range(n):
        filename = os.path.join(path_winds, "rejoin.{}".format(i))
        skip = 1 if i==0 else 0
        dfnew = pd.read_csv(filename, sep='\s+', skiprows=skip,
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
