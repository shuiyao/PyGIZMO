import pandas as pd
import warnings
import os
import utils

DefaultSchema = utils.load_default_schema()

def read_sovcirc(path_sovcirc, schema=DefaultSchema):
    df = pd.read_csv(path_sovcirc, sep='\s+', skiprows=1,
                     names=schema['sovcirc']['columns'],
                     dtype=schema['sovcirc']['dtypes'])
    df = df[:-1].set_index("haloId")
    return df

def read_stat(path_stat, schema=DefaultSchema):
    df = pd.read_csv(path_stat, sep='\s+', header=None,
                     names=schema['stat']['columns'],
                     dtype=schema['stat']['dtypes'])
    df = df.set_index("galId")
    return df

def read_sogrp(path_sogrp, schema=DefaultSchema, n_gas=None, gheader=None, gas_only=False):
    hids = pd.read_csv(path_sogrp, sep='\s+', skiprows=1,
                       names=schema['sogrp']['columns'],
                       dtype=schema['sogrp']['dtypes'])
    if(gas_only):
        if(n_gas is not None):
            ngas = n_gas
            return hids.head(ngas)
        else:
            try:
                ngas = gheader.attrs['NumPart_Total'][0]
                return hids.head(ngas)
            except:
                warnings.warn("gas_only is True but can not infer the number of gas particles.")
    return hids

def read_grp(path_grp, schema=DefaultSchema, gheader=None, n_gas=None, gas_only=False):
    gids = pd.read_csv(path_grp, sep='\s+', skiprows=1,
                       names=schema['grp']['columns'],
                       dtype=schema['grp']['dtypes'])
    if(gas_only):
        if(n_gas is not None):
            ngas = n_gas
            return gids.head(ngas)
        else:
            try:
                ngas = gheader.attrs['NumPart_Total'][0]
                return gids.head(ngas)
            except:
                warnings.warn("gas_only is True but can not infer the number of gas particles.")
    return gids

__mode__ = "test"
if __mode__ == "__test__":
    from myinit import *
    model = "l50n288-phew-m4"
    AMAX = 0.505

    snapnum = 78
    path_model = os.path.join(DIRS['DATA'], model)
    path_hdf5 = os.path.join(path_model, "snapshot_{:03d}.hdf5".format(snapnum))
    path_grp = os.path.join(path_model, "gal_z{:03d}.grp".format(snapnum))
    path_stat = os.path.join(path_model, "gal_z{:03d}.stat".format(snapnum))
    path_sogrp = os.path.join(path_model, "so_z{:03d}.sogrp".format(snapnum))
    path_sovcirc = os.path.join(path_model, "so_z{:03d}.sovcirc".format(snapnum))
    path_output = os.path.join(path_model, "snapshot_{:03d}.phewsHalos".format(snapnum))

    path_schema = os.path.join(path_model, "WINDS/initwinds.0")
    path_initwinds = os.path.join(path_model, "WINDS/initwinds.")

    halos = read_sovcirc(path_sovcirc, schema)
    hids = read_sogrp(path_sogrp, schema, gas_only=True)
    gals = read_stat(path_stat)

