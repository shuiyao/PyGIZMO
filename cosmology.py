'''
Main cosmological functions given cosmological parameters.

Set cosmological parameters in pygizmo.cfg

Cosmological parameters:
------------------------
Omega_m      : Fraction of matter in the Universe
Omega_Lambda : Fraction of dark energy in the Universe
Omega_baryon : Fraction of baryons in the Universe
hubble_param : The hubble parameter
XH           : Primodial hydrogen fraction
sigma8       : sigma8 is sigma8 lol, jargon
n            : jargon again
'''

from numpy import log10, exp, sqrt, log
from scipy import integrate
from scipy.interpolate import interp1d

from . import utils
from .astroconst import pc, ac
from .config import SimConfig


# Everything is in c.g.s units.

# Mo, Mao and White (1998):
# r200 = Vc / (10. * H(z))
# M = Vc^2 * r200 / G
#   = Vc^3 / (10. * G * H(z))
# H(z) = H0 [OmegaL + (1-Omega0-OmegaL)(1+z)^2 + Omega0(1+z)^3]^(1/2)


cfg = SimConfig()

cosmo_param = cfg.get('Cosmology')

class Cosmology():
    '''
    List of Major cosmology functions:
    --------------------------------
    info(): or help()
    all(mh, z, unit='astro'): Display all relavant info for this halo in 'astro' unit.
    H(z): Hubble parameter
    E(z): H(z) = H(0) * E(z)
    d_A(z): angular diameter distance, Mattig's formula
    Vc(mh, z, unit='c.g.s'): Circular velocity at R200. unit could be 'astro' (unit_m = Msolar)
    r200(mh, z, unit='c.g.s'): R200. unit could be 'astro' (return kpc)
    Overdensity_vir(z): Bryan & Norman 1998. ~ 100 at z = 0
    rho_crit(z): Critical density in c.g.s unit
    rhovir_over_rhobar(Omega0, z): Formula Romeel's paper
    rvir(mh, z, unit): Komatsu+ 11, Mvir = 4pi/3 r_vir^3 (Delta_c(z)*rho_crit(z))
    vvir(mh, z, unit): Vvir = sqrt(Mvir * G / Rvir)
    Tvir(mh, z, unit): Armillota+ 16, Tvir = Vvir^2 * mu * m_H / 2k
    tcosmic(a): Find the cosmic time in yr.
    acosmic(t): Find the scale factor for a cosmic time
    '''

    def __init__(self, cosmo_param):
        self._parse_cosmological_parameters(cosmo_param)
        a, tcos = utils.rcol("./data/tcosmic.dat",
                             [0, 2], linestart=1)
        self.tcosmic = interp1d(a, tcos)
        self.acosmic = interp1d(tcos, a)
        self.H0 = 1.e7 / ac.mpc * self.h

    def _parse_cosmological_parameters(self, cosmo_param):
        self.Omegam = float(cosmo_param['Omega_m'])
        self.OmegaL = float(cosmo_param['Omega_Lambda'])
        self.Omegab = float(cosmo_param['Omega_baryon'])
        self.h = float(cosmo_param['hubble_param'])
        self.XH = float(cosmo_param['XH'])
        self.sigma8 = float(cosmo_param['sigma8'])
        self.n = float(cosmo_param['n'])
    
    def E(z):
        '''
        Hubble constant in c.g.s unit.
        H(z) = H(0) * E(z)
        '''
        opz = 1. + z
        return sqrt(Omegam * opz**3 + (1.-Omegam-OmegaL)*opz**2 + OmegaL)

    def H(z):
        '''
        Hubble constant in c.g.s unit.
        H(z) = H(0) * E(z)
        '''
        return self.H0 * E(z)

    def func_da(z):
        return 1./E(z)
    def d_A(z): # angular diameter distance, Mattig's formula
        y = integrate.quad(func_da, 0., z)[0]
        y *= pc.c / Hubble0
        return y / (1. + z)

    def Vc(mh, z, unit='c.g.s'):
        if(unit == 'astro'):
            mh = mh * ac.msolar
        value = (10. * mh * pc.G * H(z))**(1./3.)
        if(unit == 'astro'): value = value / 1.e5
        return value

    def r200(mh, z, unit='c.g.s'):
        r = Vc(mh, z, unit) / (10. * H(z))
        if(unit == 'astro'): r = r * 1.e5
        if(unit == 'astro'): r = r / ac.kpc
        return r

    # Komatsu et al. (2011):
    # Mvir = 4pi/3 r_vir^3 (Delta_c(z)*rho_crit(z))
    # Bryan & Norman 1998
    def Overdensity_vir(z):
        Omegaz_minus1 = Omegam * (1. + z) ** 3 / E(z) ** 2 - 1.0
        M_PI = 3.141592653589793
        return 18.*M_PI*M_PI + 82. * Omegaz_minus1 - 39. * Omegaz_minus1 ** 2

    def rho_crit(z):
        '''
        Return the critical density at any redshift.

        Reference: Komatsu et al. (2011), WMAP Cosmology
        rho_c(z) = 2.775e11 E^2(z) h^2 M_solar Mpc^{-3}
        '''
        return ac.rhobar * E(z) ** 2

    def rhovir_over_rhobar(Omega0, z):
        M_PI = 3.141592653589793
        if(Omega0 == 1.0):
            return 178.0
        else:
            Lambda0 = 1.0 - Omega0
            x = (1. + z) ** 3
            Omegaf = Omega0*x / (Omega0*x + Lambda0)
            wf = 1./Omegaf - 1.
            answer = 18.*(M_PI*M_PI)*(1.+0.4093*(wf**0.9052))
            return answer

    def rvir(mh, z, unit='c.g.s'):
        '''
        Return the virial radius of a halo given halo mass and redshift.
        Reference: Komatsu et al. 2011
        Mvir = 4pi/3 r_vir^3 (Delta_c(z) * rho_crit(z))
        '''
        if(unit == 'astro'): mh = mh * ac.msolar
        Delta_z = Overdensity_vir(z)
        r = (mh/(4.18779*Delta_z*rho_crit(z)))**(1./3.)
        if(unit == 'astro'): r = r / ac.kpc
        return r

    def vvir(mh, z, unit='astro'):
        v = mh * pc.G / rvir(mh, z, unit=unit)
        if(unit == 'astro'):
            v = v * ac.msolar / ac.kpc
        v = sqrt(v)
        if(unit == 'astro'):
            v = v / 1.e5
        return v

    def Tvir(m, z, unit='c.g.s'): #Under SIS approximation
        '''
        Return the virial temperature of a singular isothermal sphere given halo
        mass and redshift.
        '''
        v = vvir(m, z, unit)
        if(unit == 'astro'): v = v * 1.e5
        value = v * v * 0.60 * pc.mh / (2. * pc.k)
        return value

    def rho_sis(m, z, r_rvir):
        M_PI = 3.141592653589793    
        r_vir = rvir(m, z)
        vc = Vc(m, z)
        fac = vc * vc / (4. * M_PI * pc.G)
        return fac / (r_rvir * r_vir) ** 2 / (0.60 * pc.mh)

    def halo_properties(mh, z, unit='astro'):
        '''
        Display the main properties of a galactic halo given mass of the halo and 
        the redshift.
        '''
        print ("Log(M_vir) = %7.5f [Msolar]" % (log10(mh)))
        print ("R_vir = %7.5f [kpc]" % (rvir(mh, z, unit)))
        print ("V_c = %7.5f [km/s]" % (Vc(mh, z, unit)))
        print ("T_vir = %7.5e [K] = %7.5f [keV]" % (Tvir(mh, z, unit), Tvir(mh, z, unit) * pc.k / (1.e3 * pc.eV)))

    def sis(m, z):
        '''
        Display the properties of a singlular isothermal profile given mass and
        redshift and cosmology.
        '''
        print ("Log(M_vir) = %7.5f [Msolar]" % (log10(m/ac.msolar)))
        print ("R_vir = %7.5f [kpc]" % (rvir(m, z) / ac.kpc))
        print ("V_c = %7.5f [km/s]" % (Vc(m, z) / 1.e5))
        print ("T_vir = %7.5e [K]" % (Tvir(m, z)))
        print ("n(0.1Rvir) = %7.5f [cm^-3]" % (rho_sis(m, z, 0.1)))

    
