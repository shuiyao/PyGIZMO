'''
I/O method for galaxy catelogues that were generated with SKID/SO

Example
-------
>>> from .config import SimConfig
>>> model = "l25n144-test"
>>> snapnum = 108
>>> cfg = SimConfig()

>>> path_data = os.path.join(cfg.get('Paths', 'data'), model)
>>> print("Data Folder:", path_data)
Data Folder: /home/shuiyao/codes/data/l25n144-test

>>> path_stat = os.path.join(path_data, "gal_z{:03d}.stat".format(snapnum))
>>> path_sogrp = os.path.join(path_data, "so_z{:03d}.sogrp".format(snapnum))
>>> path_sovcirc = os.path.join(path_data, "so_z{:03d}.sovcirc".format(snapnum))

>>> gals = read_stat(path_stat)
>>> print(gals.loc[:5, ['Npart','Mstar','x','y','z','Sfr']])
       Npart         Mstar         x         y         z        Sfr
galId                                                              
1        904  1.665430e-05 -0.498017 -0.342449 -0.270728  12.891000
2         13  5.245210e-08 -0.497976 -0.342248 -0.270817   0.338280
3         15  1.889210e-07 -0.498158 -0.342371 -0.270738   0.176842
4         14  2.603380e-07 -0.497970 -0.342499 -0.270611   0.000000
5         22  4.651990e-07 -0.499067 -0.344308 -0.268770   0.000000

>>> halos = read_sovcirc(path_sovcirc, DefaultSchema)
>>> halos = halos.sort_values('Mvir', ascending=False)
>>> print(halos[['Mvir','Rvir','Npart']].iloc[:5])
                Mvir        Rvir   Npart
haloId                                  
568     5.378890e+13  772.728027  225811
715     3.186370e+13  648.974976  136978
36      2.101490e+13  564.900024   86841
48      1.733190e+13  529.758972   75650
275     1.365090e+13  489.234009   56163

>>> hids = read_sogrp(path_sogrp, DefaultSchema, gas_only=True)
>>> print("Number of particles:", hids.shape[0])
Number of particles: 5663146

>>> print("Number of particles in halos:", hids.query("haloId>0").shape[0])
Number of particles in halos: 1906810

>>> print("Number of particles in central halos:", hids.query("haloId>0 and haloId==hostId").shape[0])
Number of particles in central halos: 1577315
'''

import os
import warnings
import pandas as pd

from . import utils

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

def read_sopar(path_stat, schema=DefaultSchema, as_dict=False):
    df = pd.read_csv(path_stat, sep='\s+', header=None,
                     names=schema['sopar']['columns'],
                     dtype=schema['sopar']['dtypes'])
    if(as_dict == True):
        return dict(zip(df.haloId, df.hostId))
    else:
        return df.set_index("haloId")

def read_sogrp(path_sogrp, schema=DefaultSchema, n_gas=None, gheader=None, gas_only=False):
    hids = pd.read_csv(path_sogrp, skiprows=1,
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

