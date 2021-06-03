__all__ = ['pc', 'ac']

from numpy import exp, log, pi

class pc():
    '''
    Physcial Constants in c.g.s unit.
    '''
    def __init__(self):
        self.k = 1.380658e-16
        self.c = 2.99792458e10
        self.h = 6.6260755e-27
        self.hbar = 1.05457266e-27
        self.e = 4.8032068e-10
        self.eV = self.ev = 1.6021772e-12
        self.sigma = 5.67051e-5
        self.mh = 1.6733e-24
        self.me = 9.1093897e-28
        self.G = 6.67259e-8
        self.pc = 3.08567758e18
        self.amu = self.u = 1.6605402e-24
        self.sigmat = 6.6524e-25 # Thompson scattering cross section
        self.km = 1.e5

class ac():
    '''
    Astronomical Constants in c.g.s units.
    '''
    def __init__(self):
        self.au = self.AU = 1.496e13
        self.pc = 3.08567758e18
        self.kpc = 3.08567758e21
        self.mpc = self.Mpc = 3.08567758e24
        self.msolar = 1.989e33
        self.rsolar = self.Rsolar = 6.96e10
        self.yr = 3.1536e7
        self.myr = self.yr * 1.e6
        self.gyr = self.yr * 1.e9
        self.rhobar = 3.*(100.e5 / self.mpc)**2 / (8.*pi*pc.G)

pc=pc()
ac=ac()

