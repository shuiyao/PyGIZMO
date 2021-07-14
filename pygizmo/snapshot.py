'''
The class Snapshot contains meta-data of a snapshot.
'''

import os
import warnings

import h5py

from .astroconst import pc, ac
from .config import SimConfig
from .utils import *
from . import units
from . import galaxy

from . import progen

class Snapshot(object):
    '''
    APIs on the snapshot level. 

    A snapshot is the main output from the simulation at a time-step. It 
    contains metadata of the simulation as well as attributes for each 
    individual particle at the time of the snapshot.

    Example
    -------
    >>> model = "l25n144-test"
    >>> snap = Snapshot(model, 98)
    Snapshot: l25n144-test, snapnum: 108
    >>> snap.redshift
    0.2500000073365194
    >>> snap.cosmology
    {'Omega0': 0.3, 'OmegaLambda': 0.7, 'HubbleParam': 0.7}
    >>> snap.get_units('tipsy', cgs=False).get('length')
    35714.285714285725    
    >>> snap.ngals
    1469
    >>> snap.ngas
    2561261
    >>> snap.select_galaxies_by_mass_percentiles(0.98, 0.985)    
           Npart   logMstar    logMgal
    galId                             
    48       728  10.860389  10.981041
    95       722  10.888107  11.002301
    589      673  10.881828  10.962551
    827      690  10.871061  10.991914
    841      727  10.918854  11.011083
    889      676  10.914965  10.981414
    936      653  10.907985  10.954995
    >>> snap.load_gas_particles(['PId','Mass','galId','Tmax'])
    >>> snap.gp.columns
    Index(['PId', 'Mass', 'galId', 'Tmax'], dtype='object')
    >>> snap.load_gas_particles(['PId','logT'])
    Index(['PId', 'U', 'Ne', 'Y', 'logT'], dtype='object')
    '''
    def __init__(self, model, snapnum, verbose=None, config=SimConfig()):
        self._model = model
        self._snapnum = snapnum
        self._cfg = config
        self._path_data = os.path.join(self._cfg.get('Paths','data'), model)
        self._path_hdf5 = os.path.join(self._path_data, "snapshot_{:03d}.hdf5".format(snapnum))
        self._path_workdir = os.path.join(self._cfg.get('Paths','workdir'), model)
        self._path_tmpdir = os.path.join(self._cfg.get('Paths','tmpdir'), model)
        self._path_figure = os.path.join(self._cfg.get('Paths','figure'), model)
        self.verbose = verbose

        if(not os.path.exists(self._path_workdir)):
            os.mkdir(self._path_workdir)
        if(not os.path.exists(self._path_tmpdir)):
            os.mkdir(self._path_tmpdir)
        if(not os.path.exists(self._path_figure)):
            os.mkdir(self._path_figure)
            
        self._path_grp = os.path.join(self._path_data, "gal_z{:03d}.grp".format(snapnum))
        self._path_stat = os.path.join(self._path_data, "gal_z{:03d}.stat".format(snapnum))
        self._path_sogrp = os.path.join(self._path_data, "so_z{:03d}.sogrp".format(snapnum))
        self._path_sovcirc = os.path.join(self._path_data, "so_z{:03d}.sovcirc".format(snapnum))
        self._path_sopar = os.path.join(self._path_data, "so_z{:03d}.par".format(snapnum))

        self._hdf5schema = pd.read_csv(self._cfg.get_path('Schema', 'HDF5'), header=0)\
                             .set_index('FieldName')
        with h5py.File(self._path_hdf5, "r") as hf:
            attrs = hf['Header'].attrs
            self._header_keys = set(attrs.keys())
            self._gp_keys = set(hf['PartType0'].keys())
            self._dp_keys = set(hf['PartType1'].keys())
            try:
                self._sp_keys = set(hf['PartType4'].keys())
            except:
                self._sp_keys = set()                
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

        self._boxsize_in_cm = self._boxsize * float(self._cfg.get('Units','UnitLength_in_cm'))
        self._units_tipsy = units.UnitsTipsy(
            lbox_in_mpc = self._boxsize_in_cm / ac.mpc,
            hubble_param = self._h,
            a=self._ascale)
        self._units_gizmo = units.UnitsGIZMO(
            hubble_param = self._h,
            a=self._ascale)

        self._progtable = None
        self._act = None

        talk(self.__str__(), "normal", self.verbose)

    @classmethod
    def from_file(cls, path_hdf5):
        '''
        Create an Snapshot instance directly from the HDF5 file.

        Parameters
        ----------
        path_hdf5: str.
            Complete path to the HDF5 format snapshot.
        '''
        assert(isinstance(path_hdf5, str)), "path_hdf5 must be complete path to the HDF5 file."
        assert(os.path.exists(path_hdf5)), "{} is not found.".format(path_hdf5)

        try:
            basename = os.path.basename(path_hdf5)
            suffix = os.path.splitext(basename)[-1]
            if(suffix != '.hdf5'):
                warnings.warn("path_hdf5 is likely not a HDF5 file.")
            snapstr = os.path.splitext(basename)[-2].split(sep='_')[-1]
            snapnum = int(snapstr)
            model = path_hdf5.split(sep='/')[-2]
            return cls(model, snapnum)
        except:
            raise IOError("Can not parse file path: {}.", path_hdf5)

    def __str__(self):
        return f"Snapshot: {self._model}, snapnum: {self._snapnum}"

    def __repr__(self):
        return f"Snapshot(model={self._model!r}, snapnum={self._snapnum})"

    def _get_fields_todo(self, fields, fields_exist=set()):
        if(isinstance(fields, list)): fields = set(fields)
        fields_derived = fields & set(self._cfg.get('Derived'))
        fields_derived = fields_derived ^ (fields_derived & fields_exist)
        # Need to load all the dependencies of the derived fields
        for field in fields_derived:
            fields_temp = set(self._cfg.get('Derived',field).split(sep=','))
            fields = fields | fields_temp
        fields_todrop = fields_exist ^ (fields & fields_exist)
        # Only keep the fields that is not existed
        fields = fields ^ (fields & fields_exist)
        fields_hdf5 = self._hdf5schema.index.intersection(fields)
        fields_pos = fields & set(['x', 'y', 'z'])        
        elements = self._cfg.get('Simulation','elements').split(sep=',')
        fields_metals = fields & set(elements)
        talk("Fields to be derived: {}".format(fields_derived), 'quiet')
        talk("Fields to load: {}".format(fields), 'quiet')
        return {'all':fields,
                'hdf5':fields_hdf5,
                'pos':fields_pos,                
                'metals':fields_metals,
                'derived':fields_derived,
                'todrop':fields_todrop}

    def get_units(self, system='gizmo', cgs=False):
        '''
        Get the unit system applied to this snapshot. The values depend on
        both the redshift of the snapshot and the cosmological parameters.

        Parameter
        ---------
        system: String. Default='gizmo'
            Must be either gizmo or tipsy.
        cgs: boolean. Default=False
            Display the units in cgs system? If not, display in default system.
        
        Return
        ------
        units_: Dict.
            The unit of various quantities in c.g.s units.
        '''
        assert(system in ['gizmo', 'tipsy']), "system must be either 'gizmo' or 'tipsy'"
        if(system == "gizmo"):
            units_ = self._units_gizmo.units.copy()
        if(system == "tipsy"):
            units_ = self._units_tipsy.units.copy()

        if(cgs):
            return units_
        else:
            units_sys = units.UnitsDefault().units
            for k in units_.keys():
                units_[k] = units_[k] / units_sys[k]
            return units_

    def load_gas_particles(self, fields, drop=True):
        '''
        '''
        if(isinstance(fields, str)): fields = [fields]
        if(isinstance(fields, list)): fields = set(fields)
        fields_exist = set() if self.gp is None else set(self.gp.columns)
        fields = self._get_fields_todo(fields, fields_exist)
        df = self._load_hdf5_fields('gas', fields['hdf5'], fields['pos'], fields['metals'])
        if(drop == True and fields['todrop'] != set()):
            self.gp.drop(fields['todrop'], axis=1, inplace=True)
        self.gp = df if self.gp is None else pd.concat([self.gp, df], axis=1)
        self._compute_derived_fields(fields['derived'])
        
        # Field Type: Galaxy/Halo Identifiers
        if('galId' in fields['all']):
            gids = galaxy.read_grp(self._path_grp, n_gas=self._n_gas,
                                   gas_only=True)
            self.gp = pd.concat([self.gp, gids], axis=1)
        if('haloId' in fields['all'] or 'hostId' in fields['all']):
            hids = galaxy.read_sogrp(self._path_sogrp, n_gas=self._n_gas,
                                     gas_only=True)
            # At early redshifts, no galaxy has formed yet. gids is empty.
            if(hids.empty):
                hids['haloId'] = pd.Series(np.zeros(self.ngas, dtype=int))
                hids['hostId'] = pd.Series(np.zeros(self.ngas, dtype=int))
            fields_halo = set(['haloId', 'hostId']) & fields['all']
            self.gp = pd.concat([self.gp, hids[fields_halo]], axis=1)

    def load_star_particles(self, fields, drop=True):
        '''
        '''
        if(isinstance(fields, str)): fields = [fields]
        if(isinstance(fields, list)): fields = set(fields)
        
        fields_exist = set() if self.sp is None else set(self.sp.columns)
        fields = self._get_fields_todo(fields, fields_exist)
        df = self._load_hdf5_fields('star', fields['hdf5'], fields['pos'], fields['metals'])
        if(df is None):
            return
        
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
        df = self._load_hdf5_fields('dark', ['PId'], [], [])
        hids = galaxy.read_sogrp(self._path_sogrp)
        hids = hids[self.ngas:self.ngas+self.ndark].reset_index()
        # Use the haloId and not the parentId here.
        # parentId has subsumed satellite galaxies
        self.dp = pd.concat([df, hids['haloId'], hids['hostId']], axis=1)

    def _load_hdf5_fields(self, ptype, fields_hdf5, fields_pos, fields_metals):
        '''
        '''
        if(isinstance(ptype, int)):
            assert(0 <= ptype < 6), "ptype={} is out of range [0, 6)".format(ptype)
        else:
            assert(ptype in self._cfg.get('HDF5ParticleTypes')), "ptype={} is not a valid type.".format(ptype)
            ptype = int(self._cfg.get('HDF5ParticleTypes',ptype))
        
        cols = {}
        elements = self._cfg.get('Simulation','elements').split(sep=',')        
        with h5py.File(self._path_hdf5, "r") as hf:
            try:
                hdf5part = hf['PartType'+str(ptype)]
            except:
                talk('Ignore non-existent PartType{}'.format(ptype), 'quiet')
                return
            for field in fields_hdf5:
                hdf5field = self._hdf5schema.loc[field].HDF5Field
                dtype = self._hdf5schema.loc[field].PandasType
                if(hdf5field not in hdf5part):
                    raise RuntimeError("{} is not found in the HDF5 file.".format(hdf5field))
                cols[field] = hdf5part[hdf5field][:].astype(dtype)
            # Now extract the metal field
            hdf5field = self._hdf5schema.loc['Metals'].HDF5Field
            dtype = self._hdf5schema.loc['Metals'].PandasType
            for field in fields_metals:
                cols[field] = hdf5part[hdf5field][:,elements.index(field)].astype(dtype)
            # Now extract the pos field
            posidx = {'x':0, 'y':1, 'z':2}
            hdf5field = self._hdf5schema.loc['Pos'].HDF5Field
            dtype = self._hdf5schema.loc['Pos'].PandasType
            for field in fields_pos:
                cols[field] = hdf5part[hdf5field][:,posidx[field]].astype(dtype)
        return pd.DataFrame(cols)
            
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
        fields_pos = set()
        if(fields is not None):
            fields = set(fields)
            fields_pos = fields & set(['x','y','z'])
            fields = fields ^ (fields & fields_pos)
            to_keep = fields & set(self.halos.columns)
            if(fields ^ to_keep != set()):
                warnings.warn('These fields are not found in the .sovcirc file: {}'.
                              format(fields ^ to_keep))
            self.halos = self.halos[to_keep]
        if(log_mass == True):
            for field in self.halos.columns.intersection(set(['Mvir', 'Msub'])):
                self.halos['log'+field] = np.log10(self.halos[field] / self._h)
                self.halos.drop(field, axis=1, inplace=True)
        if(fields_pos is not None):
            gals = galaxy.read_stat(self._path_stat)
            gals = gals[fields_pos]
            # gals = pd.DataFrame({'x':gals.xbound, 'y':gals.ybound, 'z':gals.zbound})
            self.halos = pd.concat([self.halos, gals], axis=1)
        self._n_gals = self.halos.shape[0]
        talk('Load {} halos ...'.format(self._n_gals), 'quiet')

    def load_phase_diagram(self, ncells_x=256, ncells_y=256, ions=False, overwrite=False):
        '''
        Build/Load 2D phase diagram from the simulation data.

        Parameters
        ----------
        ncells_x: int. Default=256
            Number of cells on the x-axis (log density)
        ncells_y: int. Default=256
            Number of cells on the y-axis (log temperature)
        ions: boolean. Default=False
            Whether or not to compute the ion fractions.
        overwrite: boolean. Default=False

        Return
        ------
        df: pandas.DataFrame.

        Outputs
        -------
        If ions==True, tabion_???.csv; else tabmet_???.csv
        '''

        from . import C

        fprefix = "tabion" if (ions) else "tabmet"
        fout = os.path.join(self._path_workdir,
                            "{}_{:03d}.csv".format(fprefix, self.snapnum))
        if(os.path.exists(fout) and overwrite==False):
            talk("Load existing phase diagram.", "normal")
            return pd.read_csv(fout)
        
        C.cpygizmo.build_phase_diagram(
            C.c_char_p(self._path_data.encode('utf-8')),
            C.c_char_p(self._path_workdir.encode('utf-8')),            
            C.c_int(self.snapnum),
            C.c_int(ncells_x),
            C.c_int(ncells_y))
        return pd.read_csv(fout)

    def load_progtable(self, reload_table=False):
        if(self._progtable is None or reload_table):
            progtable = progen.find_all_previous_progenitors(self)
            self._progtable = progtable
        else:
            talk("progtable already loaded for {}".format(self.__repr__()))
            return self._progtable
        return progtable
        
    def build_progtable(self, rebuild=False, load_halo_mass=True):
        '''
        Find the progenitors for all halos within a snapshot in all previous 
        snapshots. Calls progen.find_all_previous_progenitors(snap, overwrite, 
        load_halo_mass)

        Parameters
        ----------
        snap: class Snapshot.
        rebuild: boolean. Default=False.
            If False, first try to see if a table already exists. Create a new 
            table if not.
            If True, create a new table and overwrite the old one if needed.
        load_halo_mass: boolean. Default=True
            If True. Load logMvir and logMsub for each progenitor.

        Returns
        -------
        progtable: pandas DataFrame.
            A pandas table storing information for the progenitors of halos at
            different time.
            columns: haloId*, snapnum, progId, hostId, logMvir, logMsub

        Examples
        --------
        >>> snap = snapshot.Snapshot('l12n144-phew', 100)
        >>> progtable = snap.build_progtable(rebuild=True)

        '''
        progtable = progen.find_all_previous_progenitors(self, overwrite=rebuild, load_halo_mass=load_halo_mass)
        self._progtable = progtable
        return progtable

    def get_star_history_for_galaxy(self, galIdTarget):
        # Look at the current star particles within a galaxy
        pass

    def get_accretion_stats_for_galaxy(self, galIdTarget):
        # Look at recent accretion onto the galaxy
        pass

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

    def _transform_coordinates(self, x):
        '''
        Transform between Tipsy coordinates and GIZMO coordinates
        '''
        return (x + 0.5) * self.boxsize

    def get_phew_particles_from_halos(self, hids):
        '''
        Fetch all PhEW particles in a snapshot that appear in any of the halo in 
        a list of halos.

        Parameter
        ---------
        hids: array type.
            A list of haloIds in the snapshot.

        Returns
        -------
        phewp: pandas.DataFrame.
            Columns: haloId, snapnum, PId.
            A list of PhEW partcles that appear in the halos.

        >>> hids = pd.Series([534, 584, 374])
        '''
        if(isinstance(hids, list)): hids = pd.Series(hids)
        hids = hids[hids>0]
        self.load_gas_particles(['PId','haloId','Mc'], drop=False)
        pid = hids.apply(lambda x : list(self.gp.query("haloId==@x and Mc>0").PId))
        phewp = pd.DataFrame({'haloId':hids, 'snapnum':self.snapnum, 'PId':pid})
        # Need to remove halos that do not have any PhEW particles
        phewp = phewp.explode('PId').dropna()
        talk("{} PhEW particles fetched for {} halos.".format(phewp.shape[0], hids.size), 'talky')
        return phewp

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

    def get_gas_particles_in_galaxy(self, galId):
        '''
        Get a list of particle IDs for all gas particles that belong to a given
        galaxy. 

        Parameters
        ----------
        galId: int.
            The unique Id of the halo.

        Returns
        -------
        pIdlist: list.
        '''

        self.load_gas_particles(['PId','galId'])
        gp = self.gp.query('galId == @galId')
        talk("{} gas particles loaded from galaxy #{}".format(gp.shape[0], galId), 'normal')
        return list(gp.PId)

    def get_star_particles_in_galaxy(self, galId):
        '''
        Get a list of particle IDs for all star particles that belong to a given
        galaxy. 

        Parameters
        ----------
        galId: int.
            The unique Id of the halo.

        Returns
        -------
        pIdlist: list.
        '''

        self.load_star_particles(['PId','galId'])
        sp = self.sp.query('galId == @galId')
        talk("{} star particles loaded from galaxy #{}".format(sp.shape[0], galId), 'normal')
        return list(sp.PId)

    def get_gas_particles_in_halo(self, haloId, include_ism=False):
        '''
        Get a list of particle IDs for all gas particles that belong to a given
        halo. 

        Parameters
        ----------
        haloId: int.
            The unique Id of the halo.
        include_ism: boolean. Default=False
            If True, include the ISM particles in the list.

        Returns
        -------
        pIdlist: list.
        '''

        self.load_gas_particles(['PId','haloId'])
        if(include_ism == False):
            self.load_gas_particles(['Sfr'], drop=False)
            gp = self.gp.query('haloId == @haloId and Sfr == 0')
        else:
            gp = self.gp.query('haloId == @haloId')
        talk("{} gas particles loaded from halo #{}".format(gp.shape[0], haloId), 'normal')
        return list(gp.PId)

    def select_halos_by_mass(self, logm_min, logm_max, index_only=False, central_only=True):
        '''
        Select galaxies whose mass (logMgal) is within a percentile range.
        
        Parameters
        ----------
        logm_min: float
            Lower limit of the halo mass.
        logm_max: float
            Higher limit of the halo mass.
        index_only: boolean. Default=False
            If True, return only the indices of selected galaxies.
        central_only: boolean. Default=True
            If True, return only central halos

        Return
        ------
        pandas.Index or pandas.DataFrame.
        '''

        assert(logm_min <= logm_max), "logm_min must be smaller than logm_max"

        self.load_halos(['Msub','Mvir'])

        if(central_only):
            halos = self.halos.query("logMsub <= @logm_max and logMsub >= @logm_min")
        else:
            halos = self.halos.query("logMvir <= @logm_max and logMvir >= @logm_min")
        talk("{} halos found.".format(halos.shape[0]), "normal")
        if(index_only):
            return halos.Index()
        else:
            return halos

    def select_galaxies_by_mass(self, logm_min, logm_max, index_only=False):
        '''
        Select galaxies whose mass (logMgal) is within a percentile range.
        
        Parameters
        ----------
        logm_min: float
            Lower limit of the galaxy mass.
        logm_max: float
            Higher limit of the galaxy mass.
        index_only: boolean. Default=False
            If True, return only the indices of selected galaxies.

        Return
        ------
        pandas.Index or pandas.DataFrame.
        '''

        assert(logm_min <= logm_max), "logm_min must be smaller than logm_max"

        if(index_only):
            self.load_galaxies(['Mgal'])
        else:
            self.load_galaxies(['Npart','Mgal','Mstar'])
        gals = self.gals.query("logMgal <= @logm_max and logMal >= @logm_min")
        talk("{} galaxies found.".format(gals.shape[0]), "normal")
        if(index_only):
            return gals.Index()
        else:
            return gals

    def select_galaxies_by_mass_percentiles(self, plow, phigh, index_only=False):
        '''
        Select galaxies whose mass (logMgal) is within a percentile range.
        
        Parameters
        ----------
        plow: float, [0.0, 1.0]
            Lower limit of the percentiles.
        phigh: float, [0.0, 1.0]
            Higher limit of the percentiles.
        index_only: boolean. Default=False
            If True, return only the indices of selected galaxies.

        Return
        ------
        pandas.Index or pandas.DataFrame.
        '''

        assert(0.0 <= plow <= 1.0), "plow must be within [0.0, 1.0]."
        assert(0.0 <= phigh <= 1.0), "phigh must be within [0.0, 1.0]."
        assert(plow <= phigh), "plow must be smaller than phigh"

        if(index_only):
            self.load_galaxies(['Mgal'])
        else:
            self.load_galaxies(['Npart','Mgal','Mstar'])
        mlower = self.gals.logMgal.quantile(plow)        
        mupper = self.gals.logMgal.quantile(phigh)
        gals = self.gals.query("logMgal <= @mupper and logMgal >= @mlower")
        talk("{} galaxies found.".format(gals.shape[0]), "normal")        
        if(index_only):
            return gals.Index()
        else:
            return gals
    
def _test(model=None):
    model = "l25n144-test" if (model is None) else model
    snap = Snapshot(model, 98)

    # Check cosmology and redshift
    snap.cosmology
    print(f"redshift = {snap.redshift}")

    # Check internal units:
    unit_l = snap.get_units('tipsy', cgs=False).get('length')
    print(f"Unit Length in TIPSY format: {unit_l}")

    print(f"Number of gas particles: {snap.ngas}")

    print(f"Number of galaxies: {snap.ngals}")

    # Find the galaxies whose mass is within the top 1.5% to2%
    print(snap.select_galaxies_by_mass_percentiles(0.98, 0.985))

    # load PId, Mass, Tmax from HDF5 snapshot file
    # load galId from galaxy CSV file
    snap.load_gas_particles(['PId','Mass','galId','Tmax'])
    print(snap.gp.columns)
    # derive logT from U, Ne, Y fields
    snap.load_gas_particles(['PId','logT'])
    print(snap.gp.columns)
    
if(__name__ == "__main__"):
    _test(model)
