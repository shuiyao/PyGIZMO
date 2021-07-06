'''
Define the unit system to use.
'''

from astroconst import pc, ac
from numpy import exp, log, pi, sqrt
import configparser
from config import SimConfig
import abc

class Units(object):
    '''
    Example
    -------
    u_tipsy = UnitsTipsy(lbox_in_mpc=50)
    u_astro = UnitsDefault()
    mass_physical = mass_tipsy * u_tipsy.m / u_astro.m
    '''
    
    def __init__(self, config=SimConfig()):
        self._cfg = config
        self.l = 1.0
        self.m = 1.0
        self.t = 1.0
        self.d = 1.0
        self.v = 1.0
        self.u = 1.0
        self.b = 1.0
        self._units = {}
        self._update_units()

    def _update_units(self):
        self._units['length'] = self.l
        self._units['mass'] = self.m
        self._units['time'] = self.t
        self._units['velocity'] = self.v
        self._units['density'] = self.d
        self._units['energy'] = self.u
        self._units['magneticfield'] = self.b

    def keys(self):
        return ['length', 'mass', 'time', 'velocity', 'density',
                'energy', 'magneticfield']

    def get_unit(self, unitstr):
        '''
        Get the unit specified in the unitstr.

        Parameters
        ----------
        unitstr: String.
            One of ['length', 'mass', 'time', 'velocity', 'density', 'energy',
            'magneticfield'].
        '''

        if(unitstr not in self.keys()):
            raise KeyError("Supported unit strings: {}".format(self.keys()))
        
        return self._units[unitstr]

    def convert(self, arr, unitstr):
        '''
        Convert an array of values in the current unit system to c.g.s.

        Parameters
        ----------
        arr: ArrayLike.
            The input array.
        unitstr: String.
            One of ['length', 'mass', 'time', 'velocity', 'density', 'energy',
            'magneticfield'].
        '''

        unit = self.get_unit(unitstr)
        return arr * unit

    @property
    def units(self):
        return self._units

class UnitsDefault(Units):
    def __init__(self, config=SimConfig()):
        super(UnitsDefault, self).__init__(config)

        system_units = self._cfg.get('Units')
        # The system unit is NOT a self-consistent system
        self.l = float(system_units['SystemLength'])        # 1 kpc
        self.m = float(system_units['SystemMass'])          # 1 Msolar
        self.t = float(system_units['SystemTime'])          # 1 Myr
        self.d = float(system_units['SystemDensity'])       # H/cm^3
        self.v = float(system_units['SystemVelocity'])      # 1 km/s
        self.u = float(system_units['SystemEnergy'])        # 10000 K (XH=0.76)
        self.b = float(system_units['SystemMagneticField']) # 1 Gauss

class UnitsComoving(Units):
    def __init__(self, hubble_param, a=None, z=None, config=SimConfig()):
        '''
        Parameters
        ----------
        hubble_param: float. Default: 0.7
          The hubble constant of the Universe is hubble_param * 100 km/s/Mpc
        a: float. Default: 1.0
          The scale factor of the Universe. Related to redshift as a = 1/(1+z)
        lbox_in_mpc: float. Default: -1
          The box length in Mpc.
          Required for unit conversion with the tipsy system.
        '''

        super(UnitsComoving, self).__init__(config)
        
        self._hubble_param = hubble_param
        self._validate_input(a=a, z=z)
        self.a = 1./(1.+z) if (a is None) else a
        self.z = 1./a - 1. if (z is None) else z
        self._comoving = True

    @staticmethod
    def _validate_input(a=None, z=None):
        if(a is not None and z is not None):
            raise ValueError("Can only take either a or z but not both.")
        if(a is None and z is None):
            raise ValueError("Must set either a or z.")

        if(z is None):
            try:
                a = float(a)
            except:
                raise TypeError("One must be able to cast a into float.")
            if(a <= 0.0 or a > 1.0):
                raise ValueError(f"a = {a} is out of bounds 0 < a <= 1.")
        if(a is None):
            try:
                z = float(z)
            except:
                raise TypeError("One must be able to cast z into float.")
            if(z < 0.0):
                raise ValueError(f"z = {z} is out of bounds z >= 0")

    def update_atime(self, a):
        self.__class__._validate_input(a)
        self.comoving_to_physical()
        self.physical_to_comoving(a)
        self.a = a
        self.z = 1./a - 1.

    def update_redshift(self, z):
        self.__class__._validate_input(z)
        self.update_atime(1./(z + 1.))

    def comoving_to_physical():
        if(not self._comoving):
            return
        self.l = self.l * self.a
        self.v = self.v * sqrt(self.a)
        self.d = self.d / (self.a ** 3)
        self._comoving = False
        self._update_units()        

    def physical_to_comoving():
        if(self._comoving):
            return
        self.l = self.l / self.a
        self.v = self.v / sqrt(self.a)
        self.d = self.d * (self.a ** 3)
        self._comoving = True
        self._update_units()

    @property
    def hubble_param(self):
        return self._hubble_param

    @property
    def comoving(self):
        return self._comoving

    
class UnitsGIZMO(UnitsComoving):
    '''
    Unit definition in GIZMO:

    h = H0 / (100 km/s/Mpc) = HubbleParam = (params.txt default = 0.7)  
    MASS_code = UnitMass_in_g / h = (params.txt default = 10^10 h^-1 M_sun)  
    LENGTH_code = UnitLength_in_cm / h = (params.txt default = 1 h^-1 kpc)  
    VELOCITY_code = UnitVelocity_in_cm_per_s = (params.txt default = 1 km/s)  
    TIME_code = LENGTH_code/VELOCITY_code = (params.txt default = 0.978 h^-1 Gyr)
    INTERNAL ENERGY_code = VELOCITY_code^2 = (params.txt default = (km/s)^2)  
    DENSITY_code = MASS_code/(LENGTH_code^3) = (params.txt default = 6.77e-22 h^-2 g/cm^3)  
    MAGNETIC_FIELD_code = UnitMagneticField_in_gauss = (params.txt default = 1 Gauss)  
    DIVERGENCE_DAMPING_FIELD_code = MAGNETIC_FIELD_code * VELOCITY_code

    a_scale = 1/(1+z) = scale factor, also the "time unit" of the code  
    LENGTH_physical = LENGTH_code * a_scale  
    MASS_physical = MASS_code  
    VELOCITY_physical = VELOCITY_code * sqrt(a_scale)  
    DENSITY_physical = DENSITY_CODE / a_scale^3  
    KERNEL_LENGTH_physical = KERNEL_LENGTH_CODE * a_scale  
    MAGNETIC_FIELD_physical = MAGNETIC_FIELD_code  
    INTERNAL_ENERGY_physical = INTERNAL_ENERGY_code  
    MAGNETIC_FIELD_physical = MAGNETIC_FIELD_code (note that *in* the code, co-moving units are used B_code=a_scale^2 B_phys, but in outputs these are converted to physical)  
    DIVERGENCE_DAMPING_FIELD_physical = DIVERGENCE_DAMPING_FIELD_code (again, in-code, psi_physical = a_scale^3 psi_code)

    http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-units
    '''
    
    def __init__(self, hubble_param, a=None, z=None):
        '''
        Parameters
        ----------
        hubble_param: float.
          The hubble constant of the Universe is hubble_param * 100 km/s/Mpc
        a: float. Default: 1.0
          The scale factor of the Universe. Related to redshift as a = 1/(1+z)
        '''

        if(a is None and z is None):
            a = 1.0
        super(UnitsGIZMO, self).__init__(hubble_param=hubble_param, a=a, z=z)

        # Gadget Units in comoving c.g.s
        self.l = float(self._cfg.get("Units",'UnitLength_in_cm'))
        self.m = float(self._cfg.get("Units",'UnitMass_in_g'))
        self.v = float(self._cfg.get("Units",'UnitVelocity_in_cm_per_s'))
        self.b = float(self._cfg.get("Units",'UnitMagneticField_in_gauss'))
        self.d = self.m / self.l ** 3
        self.t = self.l / self.v # 3.086e16 s ~ 0.97 Gyr
        self.u = self.u * self.u # (km/s **2)

        self.l = self.l / self.hubble_param
        self.t = self.t / self.hubble_param
        self.m = self.m / self.hubble_param
        self.d = self.d / (self.hubble_param ** 2)

        self._update_units()

class UnitsTipsy(UnitsComoving):
    def __init__(self, lbox_in_mpc, hubble_param, a=None, z=None):
        '''
        Parameters
        ----------
        hubble_param: float.
          The hubble constant of the Universe is hubble_param * 100 km/s/Mpc
        a: float. Default: 1.0
          The scale factor of the Universe. Related to redshift as a = 1/(1+z)
        lbox_in_mpc: float.
          The box length in Mpc.
          Required for unit conversion with the tipsy system.
        '''

        if(a is None and z is None):
            a = 1.0
        
        super(UnitsTipsy, self).__init__(hubble_param, a=a, z=z)
        assert (lbox_in_mpc > 0), "Have to set lbox_in_mpc in the 'tipsy' unit system!"

        self._boxsize = lbox_in_mpc

        self.l = lbox_in_mpc * ac.mpc / self.hubble_param
        self.d = ac.rhobar # H0 = 100 km/s/Mpc
        self.t = (self.d * pc.G)**(-0.5) / (self.hubble_param * self.hubble_param)
        # (8.*pi/3.)**(0.5)*ac.mpc / (100.*hubble_param**2*UNIT_GADGET_V)
        self.m = self.d * self.l**3 * (self.hubble_param ** 2)
        self.v = self.l / self.t
        self.u = self.v * self.v
        self.b = 1.0

        self._update_units()

    @property
    def boxsize(self):
        return self._boxsize
