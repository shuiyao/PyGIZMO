'''
Basic loading of the fast C library.
'''
from .. import environment
from ctypes import cdll, c_void_p, c_size_t, c_double, c_int, c_uint, POINTER, \
                   byref, create_string_buffer, c_char_p
cpygad = cdll.LoadLibrary(environment.module_dir+'C/cpygad.so')

cpygad.cubic.restype = c_double
cpygad.cubic.argtypes = [c_double, c_double]
cpygad.quartic.restype = c_double
cpygad.quartic.argtypes = [c_double, c_double]
cpygad.quintic.restype = c_double
cpygad.quintic.argtypes = [c_double, c_double]
cpygad.Wendland_C2.restype = c_double
cpygad.Wendland_C2.argtypes = [c_double, c_double]
cpygad.Wendland_C4.restype = c_double
cpygad.Wendland_C4.argtypes = [c_double, c_double]
cpygad.Wendland_C6.restype = c_double
cpygad.Wendland_C6.argtypes = [c_double, c_double]

cpygad.Voigt.restype = c_double
cpygad.Voigt.argtypes = [c_double, c_double, c_double]

