import os
import configparser
from . import utils

class SimConfig():
    '''
    APIs for the configuration file.

    Parameters
    ----------
    path_config: String.
        The path to the configuration file.
        If None, use the default pygizmo.cfg file.

    Example
    -------
    >>> cfg = SimConfig()

    >>> cfg.sections()
    ['DEFAULT', 'Paths', 'Schema', 'Verbose', 'Units', 'Cosmology', 'Default', 'Simulation', 'Ions', 'Zsolar', 'HDF5Fields', 'HDF5ParticleTypes', 'Derived']

    >>> cfg.keys('Simulation')
    ['snapnum_reference', 'n_metals', 'elements']

    >>> cfg.get('Simulation', 'elements')
    'Z,Y,C,N,O,Ne,Mg,Si,S,Ca,Fe'

    >>> cfg.get_path('Schema', 'HDF5')
    '/home/shuiyao/codes/pygizmo/pygizmo/data/HDF5schema.csv'
    '''

    def __init__(self, path_config=None):
        if(path_config is None):
            self._path = os.path.join(os.path.dirname(__file__),
                                      "pygizmo.cfg")
        else:
            self._path = path_config
        self._cfg = self.load_config_file()

    def load_config_file(self, path_config=None):
        if(path_config is not None):
            self._path = path_config
        tmp = configparser.ConfigParser(inline_comment_prefixes=('#'))
        tmp.optionxform=str
        tmp.read(self._path)
        return tmp

    def sections(self):
        print(list(self._cfg))

    def keys(self, category):
        section = self._cfg[category]
        print(list(section))

    def get(self, category, field=None, dtype=None):
        cfg = self._cfg[category]
        if(field is not None):
            cfg = cfg[field]
        else:
            return cfg
        return cfg

    def get_path(self, category, field):
        relpath = self.get(category, field)
        return utils.abspath(relpath)
