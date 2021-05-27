#from .astroconst import pc, ac
from astroconst import pc, ac
from numpy import exp, log, pi, sqrt
import configparser

# Gadget Units in comoving c.g.s
cfg = configparser.ConfigParser(inline_comment_prefixes=('#'))
cfg.read('pygizmo.cfg')
UNITS = cfg['UNITS']

UNIT_GADGET_L = float(UNITS['UnitLength_in_cm'])
UNIT_GADGET_M = float(UNITS['UnitMass_in_g'])
UNIT_GADGET_V = float(UNITS['UnitVelocity_in_cm_per_s'])
UNIT_GADGET_B = float(UNITS['UnitMagneticField_in_gauss'])
UNIT_GADGET_D = UNIT_GADGET_M / UNIT_GADGET_L ** 3
UNIT_GADGET_T = UNIT_GADGET_L / UNIT_GADGET_V # 3.086e16 s ~ 0.97 Gyr
UNIT_GADGET_U = UNIT_GADGET_V * UNIT_GADGET_V # (km/s **2)

class Units(object):
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

    Parameters
    ----------
    system: string. Default: default
      The unit system. Must be one of ['default', 'gadget', 'tipsy', 'cgs']
    hubble_param: float. Default: 0.7
      The hubble constant of the Universe is hubble_param * 100 km/s/Mpc
    a: float. Default: 1.0
      The scale factor of the Universe. Related to redshift as a = 1/(1+z)
    lbox_in_mpc: float. Default: -1
      The box length in Mpc.
      Required for unit conversion with the tipsy system.
    comoving: bool. Default: False
      If True, keep units in comoving system.

    Example
    -------
    u_tipsy = Units('tipsy', lbox_in_mpc=50)
    u_astro = Units('default')
    mass_physical = mass_tipsy * u_tipsy.m / u_astro.m
    '''
    def __init__(self, system="default", hubble_param=0.7, a=1.0, lbox_in_mpc=-1, comoving=False):
        self.a = a
        self.hubble_param = hubble_param
        if(system == 'default'):
            # Not a consistent system
            self.l = ac.kpc
            self.t = ac.myr
            self.m = ac.msolar
            self.d = pc.mh # m_H / cm**3
            self.v = 1.0e5 # km/
            self.u = pc.k * 1.e4 / pc.mh # ~ 1.e4 K
            self.b = 1.0
        elif(system == 'cgs'):
            self.l = 1.0
            self.t = 1.0
            self.m = 1.0
            self.d = 1.0
            self.v = 1.0
            self.u = 1.0
            self.b = 1.0            
        elif(system == 'gadget'):
            self.l = UNIT_GADGET_L / hubble_param
            self.t = UNIT_GADGET_T / hubble_param
            self.m = UNIT_GADGET_M / hubble_param
            self.d = UNIT_GADGET_D / (hubble_param ** 2)
            self.v = UNIT_GADGET_V
            self.u = UNIT_GADGET_U
            self.b = UNIT_GADGET_B
        elif(system == 'tipsy'):
            assert (lbox_in_mpc > 0), "Have to set lbox_in_mpc in the 'tipsy' unit system!"
            self.l = lbox_in_mpc * ac.mpc / hubble_param
            self.d = ac.rhobar # H0 = 100 km/s/Mpc
            self.t = (self.d * pc.G)**(-0.5) / (hubble_param * hubble_param)
            # (8.*pi/3.)**(0.5)*ac.mpc / (100.*hubble_param**2*UNIT_GADGET_V)
            self.m = self.d * self.l**3 * (hubble_param ** 2)
            self.v = self.l / self.t
            self.u = self.v * self.v
            self.b = 1.0
        else:
            raise ValueError("Argument 'system' must be one of ['default', 'gadget', 'tipsy', 'cgs']")
        if not comoving:
            self._convert_comoving_to_physical(a)

    def _convert_comoving_to_physical(self, a):
        self.l = self.l * a
        self.m = self.m
        self.v = self.v * sqrt(a)
        self.d = self.d / (a ** 3)
        self.u = self.u
        self.t = self.t
        self.b = self.b
