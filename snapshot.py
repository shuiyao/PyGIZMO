'''
The class Snapshot contains meta-data of a snapshot.
'''

import h5py
import os
import numpy as np
import pandas as pd
import pdb
import galaxy
import warnings

import units
from astroconst import pc, ac

import config
from utils import talk
cfg = config.cfg

path_schema = "HDF5schema.csv"
hdf5schema = pd.read_csv(path_schema, header=0).set_index('FieldName')
PATHS = cfg['Paths']

class Snapshot(object):
    def __init__(self, model, snapnum):
        self._model = model
        self._snapnum = snapnum
        self._path_model = os.path.join(PATHS['data'], model)
        self._path_hdf5 = os.path.join(self._path_model, "snapshot_{:03d}.hdf5".format(snapnum))
        self._path_workdir = os.path.join(PATHS['workdir'], model)
        if(not os.path.exists(self._path_workdir)):
            os.mkdir(self._path_workdir)
        self._path_grp = os.path.join(self._path_model, "gal_z{:03d}.grp".format(snapnum))
        self._path_stat = os.path.join(self._path_model, "gal_z{:03d}.stat".format(snapnum))
        self._path_sogrp = os.path.join(self._path_model, "so_z{:03d}.sogrp".format(snapnum))
        self._path_sovcirc = os.path.join(self._path_model, "so_z{:03d}.sovcirc".format(snapnum))
        self._path_sopar = os.path.join(self._path_model, "so_z{:03d}.par".format(snapnum))        

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
            self._h = self._cosmology['HubbleParam']
        self.gp = None
        self.dp = None
        self.sp = None
        self.gals = None
        self.halos = None
        self._n_gals = None

        self._boxsize_in_cm = self._boxsize * float(cfg['Units']['UnitLength_in_cm'])
        self._units_tipsy = units.Units('tipsy', lbox_in_mpc = self._boxsize_in_cm / ac.mpc)
        self._units_gizmo = units.Units('gadget')

    @classmethod
    def from_file(path_hdf5):
        '''
        Create an Snapshot instance directly from the HDF5 file.
        '''
        pass

    def _get_fields_todo(self, fields, fields_exist=set()):
        if(isinstance(fields, list)): fields = set(fields)
        fields_derived = fields & set(cfg['Derived'])
        fields_derived = fields_derived ^ (fields_derived & fields_exist)
        # Need to load all the dependencies of the derived fields
        for field in fields_derived:
            fields_temp = set(cfg['Derived'][field].split(sep=','))
            fields = fields | fields_temp
        fields_todrop = fields_exist ^ (fields & fields_exist)
        # Only keep the fields that is not existed
        fields = fields ^ (fields & fields_exist)
        fields_hdf5 = hdf5schema.index.intersection(fields)
        elements = cfg['Simulation']['elements'].split(sep=',')
        fields_metals = fields & set(elements)
        talk("Fields to be derived: {}".format(fields_derived), 'quiet')
        talk("Fields to load: {}".format(fields), 'quiet')
        return {'all':fields,
                'hdf5':fields_hdf5,
                'metals':fields_metals,
                'derived':fields_derived,
                'todrop':fields_todrop}

    def load_gas_particles(self, fields, drop=True):
        '''
        '''
        if(isinstance(fields, str)): fields = [fields]
        if(isinstance(fields, list)): fields = set(fields)
        fields_exist = set() if self.gp is None else set(self.gp.columns)
        fields = self._get_fields_todo(fields, fields_exist)
        df = self._load_hdf5_fields('gas', fields['hdf5'], fields['metals'])
        if(drop == True and fields['todrop'] != set()):
            self.gp.drop(fields['todrop'], axis=1, inplace=True)
        self.gp = df if self.gp is None else pd.concat([self.gp, df], axis=1)
        self._compute_derived_fields(fields['derived'])
        
        # Field Type: Galaxy/Halo Identifiers
        if('galId' in fields['all']):
            gids = galaxy.read_grp(self._path_grp, n_gas=self._n_gas, gas_only=True)
            self.gp = pd.concat([self.gp, gids], axis=1)
        if('haloId' in fields['all'] or 'hostId' in fields['all']):
            hids = galaxy.read_sogrp(self._path_sogrp, n_gas=self._n_gas, gas_only=True)
            fields_halo = set(['haloId', 'hostId']) & fields['all']
            self.gp = pd.concat([self.gp, hids[fields_halo]], axis=1)

    def load_star_particles(self, fields, drop=True):
        '''
        '''
        if(isinstance(fields, str)): fields = [fields]
        if(isinstance(fields, list)): fields = set(fields)
        fields_exist = set() if self.sp is None else set(self.sp.columns)
        fields = self._get_fields_todo(fields, fields_exist)
        df = self._load_hdf5_fields('star', fields['hdf5'], fields['metals'])
        if(drop == True and fields['todrop'] != set()):
            self.sp.drop(fields['todrop'], axis=1, inplace=True)
        self.sp = df if self.sp is None else pd.concat([self.sp, df], axis=1)
        self._compute_derived_fields(fields['derived'])

        # Field Type: Galaxy/Halo Identifiers
        if('galId' in fields['all']):
            gids = galaxy.read_grp(self._path_grp)
            gids = gids.tail(int(self._n_star)).reset_index()
            self.sp = pd.concat([self.sp, gids['galId']], axis=1)
        if('haloId' in fields['all']):
            hids = galaxy.read_sogrp(self._path_sogrp)
            hids = hids.tail(int(self._n_star)).reset_index()
            self.sp = pd.concat([self.sp, hids['haloId']], axis=1)

    def load_dark_particles(self, force_reload=False):
        '''
        Unlike gas and stars, we mostly only care about the mass and the halo
        info of dark matter particles.
        '''
        if(self.dp is not None and force_reload == False):
            talk("Dark particles already loaded. Use force_reload to reload.", 'quiet')
            return
        df = self._load_hdf5_fields('dark', ['PId'], [])
        hids = galaxy.read_sogrp(self._path_sogrp)
        hids = hids[self.ngas:self.ngas+self.ndark].reset_index()
        # Use the haloId and not the parentId here.
        # parentId has subsumed satellite galaxies
        self.dp = pd.concat([df, hids['haloId'], hids['hostId']], axis=1)

    def _load_hdf5_fields(self, ptype, fields_hdf5, fields_metals):
        '''
        '''
        if(isinstance(ptype, int)):
            assert(0 <= ptype < 6), "ptype={} is out of range [0, 6)".format(ptype)
        else:
            assert(ptype in cfg['HDF5ParticleTypes']), "ptype={} is not a valid type.".format(ptype)
            ptype = int(cfg['HDF5ParticleTypes'][ptype])
        
        cols = {}
        elements = cfg['Simulation']['elements'].split(sep=',')        
        with h5py.File(self._path_hdf5, "r") as hf:
            hdf5part = hf['PartType'+str(ptype)]
            for field in fields_hdf5:
                hdf5field = hdf5schema.loc[field].HDF5Field
                dtype = hdf5schema.loc[field].PandasType
                if(hdf5field not in hdf5part):
                    raise RuntimeError("{} is not found in the HDF5 file.".format(hdf5field))
                cols[field] = hdf5part[hdf5field][:].astype(dtype)
            # Now extract the metal field
            hdf5field = hdf5schema.loc['Metals'].HDF5Field
            dtype = hdf5schema.loc['Metals'].PandasType
            for field in fields_metals:
                cols[field] = hdf5part[hdf5field][:,elements.index(field)].astype(dtype)
        return pd.DataFrame(cols)
            
    def _compute_derived_fields(self, fields_derived):
        cols = {}
        if('logT' in fields_derived):
            talk("Calculating for derived field: logT ...", 'quiet')
            ne = self.gp['Ne']
            u = self.gp['U']
            xhe = self.gp['Y']
            mu= (1.0 + 4.0 * xhe) / (1.0 + ne + xhe)
            # u = (3/2)kT/(mu*m_H)
            logT = np.log10(u * self._units_gizmo.u * mu * pc.mh /
                            (1.5 * pc.k))
            cols['logT'] = logT.astype('float32')
        self.gp = pd.concat([self.gp, pd.DataFrame(cols)], axis=1)

    def load_galaxies(self, fields=None, log_mass=True):
        self.gals = galaxy.read_stat(self._path_stat)
        if(fields is not None):
            fields = set(fields)
            to_keep = fields & set(self.gals.columns)
            if(fields ^ to_keep != set()):
                warnings.warn('These fields are not found in the .stat file: {}'.
                              format(fields ^ to_keep))
            self.gals = self.gals[to_keep]
        if(log_mass == True):
            for field in self.gals.columns.intersection(set(['Mgal', 'Mgas', 'Mstar'])):
                self.gals['log'+field] = np.log10(self.gals[field] * self._units_tipsy.m / ac.msolar)
                self.gals.drop(field, axis=1, inplace=True)
        self._n_gals = self.gals.shape[0]
        talk('Load {} galaxies ...'.format(self._n_gals), 'quiet')

    def load_halos(self, fields=None, log_mass=True):
        self.halos = galaxy.read_sovcirc(self._path_sovcirc)
        if(fields is not None):
            fields = set(fields)
            to_keep = fields & set(self.halos.columns)
            if(fields ^ to_keep != set()):
                warnings.warn('These fields are not found in the .sovcirc file: {}'.
                              format(fields ^ to_keep))
            self.halos = self.halos[to_keep]
        if(log_mass == True):
            for field in self.halos.columns.intersection(set(['Mvir', 'Msub'])):
                self.halos['log'+field] = np.log10(self.halos[field] / self._h)
                self.halos.drop(field, axis=1, inplace=True)
        self._n_gals = self.halos.shape[0]
        talk('Load {} halos ...'.format(self._n_gals), 'quiet')

    @property
    def model(self):
        return self._model
                
    @property
    def snapnum(self):
        return int(self._snapnum)
                
    @property
    def ngas(self):
        return int(self._n_gas)

    @property
    def ndark(self):
        return int(self._n_dark)

    @property
    def nstar(self):
        return int(self._n_star)

    @property
    def boxsize(self):
        return float(self._boxsize)

    @property
    def redshift(self):
        return float(self._redshift)

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

    @property
    def ngals(self):
        if(self._n_gals is not None):
            return int(self._n_gals)
        self.load_galaxies(log_mass=False)
        return int(self._n_gals)
    
    
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
