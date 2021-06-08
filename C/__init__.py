'''
Basic loading of the fast C library.
'''

import os
# from .. import config

# path_pygizmo = config.cfg['Paths']['pygizmo']
path_pygizmo = "/home/shuiyao_umass_edu/pygizmo/C/"

from ctypes import cdll, c_void_p, c_size_t, c_double, c_int, c_uint, c_char_p, c_wchar_p, create_string_buffer

cpygizmo = cdll.LoadLibrary(os.path.join(path_pygizmo, 'cpygizmo.so'))

