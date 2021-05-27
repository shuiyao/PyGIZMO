'''
The class Snapshot contains meta-data of a snapshot.
'''

import h5py
import os
import configparser
import pandas as pd
import pdb
import galaxy
import warnings

import units
from astroconst import pc, ac

cfg = configparser.ConfigParser(inline_comment_prefixes=('#'))
cfg.read('pygizmo.cfg')
path_schema = "HDF5schema.csv"
hdf5schema = pd.read_csv(path_schema, header=0).set_index('FieldName')
PATHS = cfg['PATHS']

class Snapshot(object):
    def __init__(self, model, snapnum):
        self._model = model
        self._path_model = os.path.join(PATHS['DATA'], model)
        self._path_hdf5 = os.path.join(self._path_model, "snapshot_{:03d}.hdf5".format(snapnum))
        self._path_output = os.path.join(PATHS['SCIDATA'], model)
        if(not os.path.exists(self._path_output)):
            os.mkdir(self._path_output)
        self._path_grp = os.path.join(self._path_model, "gal_z{:03d}.grp".format(snapnum))
        self._path_stat = os.path.join(self._path_model, "gal_z{:03d}.stat".format(snapnum))
        self._path_sogrp = os.path.join(self._path_model, "so_z{:03d}.sogrp".format(snapnum))
        self._path_sovcirc = os.path.join(self._path_model, "so_z{:03d}.sovcirc".format(snapnum))

        with h5py.File(self._path_hdf5, "r") as hf:
            attrs = hf['Header'].attrs
            self._header_keys = set(attrs.keys())
            self._gp_keys = set(hf['PartType0'].keys())
            self._dp_keys = set(hf['PartType1'].keys())
            self._sp_keys = set(hf['PartType4'].keys())
            self._n_gas = attrs['NumPart_Total'][0]
            self._n_dark = attrs['NumPart_Total'][1]
            self._n_star = attrs['NumPart_Total'][4]
            self._n_part = attrs['NumPart_Total']
            self._n_tot = sum(attrs['NumPart_Total'])
            self._boxsize = attrs['BoxSize']
            self._redshift = max(attrs['Redshift'], 0)
            self._ascale = min(attrs['Time'], 1.0)
            self._cosmology = {'Omega0':attrs['Omega0'],
                               'OmegaLambda':attrs['OmegaLambda'],
                               'HubbleParam':attrs['HubbleParam']}
        self.gp = None
        self.dp = None
        self.sp = None
        self.gals = None
        self.halos = None

        self._boxsize_in_cm = self._boxsize * float(cfg['UNITS']['unitlength_in_cm'])
        self._units_tipsy = units.Units('tipsy', lbox_in_mpc = self._boxsize_in_cm / ac.mpc)
            
    def load_gas_particles(self, fields, drop=True):
        '''
        Load fields of gas particles from various sources.
        '''
        # Only load fields that aren't already existed
        try:
            existed_fields = set(self.gp.columns)
        except:
            existed_fields = set()
        fields = set(fields)

        if(drop):
            # Drop fields that are not in 'fields'
            drop_fields = existed_fields ^ (fields & existed_fields)
            if(drop_fields != set()):
                self.gp.drop(drop_fields, axis=1, inplace=True)

        # Only keep the fields that is not existed
        fields = fields ^ (fields & existed_fields)

        fields_hdf5 = fields & set(hdf5schema.index)
        elements = cfg['SIMULATION']['elements'].split(sep=',')
        fields_Zmet = fields & set(elements)

        cols = {}
        with h5py.File(self._path_hdf5, "r") as hf:
            gp = hf['PartType0']
            for field in fields_hdf5:
                hdf5field = hdf5schema.loc[field].HDF5Field
                dtype = hdf5schema.loc[field].PandasType
                if(hdf5field not in gp):
                    raise RuntimeError("{} is not found in the HDF5 file.".format(hdf5field))
                cols[field] = gp[hdf5field][:].astype(dtype)
            # Now extract the metal field
            hdf5field = hdf5schema.loc['Zmet'].HDF5Field
            dtype = hdf5schema.loc['Zmet'].PandasType
            for field in fields_Zmet:
                cols[field] = gp[hdf5field][:,elements.index(field)].astype(dtype)

        if(self.gp is None):
            self.gp = pd.DataFrame(cols)
        else:
            self.gp = pd.concat([self.gp, pd.DataFrame(cols)], axis=1)

        if('galId' in fields):
            gids = galaxy.read_grp(self._path_grp, n_gas=self._n_gas, gas_only=True)
            self.gp = pd.concat([self.gp, gids], axis=1)

        if('haloId' in fields):
            hids = galaxy.read_sogrp(self._path_sogrp, n_gas=self._n_gas, gas_only=True)
            self.gp = pd.concat([self.gp, hids], axis=1)

    def load_galaxies(self, fields=None):
        self.gals = galaxy.read_stat(self._path_stat)
        if(fields is not None):
            fields = set(fields)
            to_keep = fields & set(self.gals.columns)
            if(fields ^ to_keep != set()):
                warnings.warn('These fields are not found in the .stat file: {}'.
                              format(fields ^ to_keep))
            self.gals = self.gals[to_keep]

    @property
    def ngas(self):
        return self._n_gas

    @property
    def ndark(self):
        return self._n_dark

    @property
    def nstar(self):
        return self._n_star

    @property
    def boxsize(self):
        return self._boxsize

    @property
    def redshift(self):
        return self._redshift

    @property
    def ascale(self):
        return self._ascale

    @property
    def cosmology(self):
        return self._cosmology

    @property
    def gp_keys(self):
        return self._gp_keys

    @property
    def sp_keys(self):
        return self._sp_keys
    
    
'''
>>> snap = Snapshot("l12n144-phew-movie-200", 100)
>>> snap.load_gas_particles(['PId','Mass'])
>>> snap.gp.columns
Index(['PId', 'Mass'], dtype='object')
>>> snap.load_gas_particles(['PId','Tmax'])
>>> snap.gp.columns
Index(['PId', 'Tmax'], dtype='object')
>>> snap.load_gas_particles(['Ne'], drop=False)
>>> snap.gp.columns
Index(['PId', 'Tmax', 'Ne'], dtype='object')
'''

snap = Snapshot("l12n144-phew-movie-200", 100)
