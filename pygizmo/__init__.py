'''
Module for units and arrays with units.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(config)
    TestResults(failed=0, attempted=5)
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=14)
    >>> doctest.testmod(cosmology)
    TestResults(failed=0, attempted=0)

    # >>> doctest.testmod(simulation)
    # TestResults(failed=0, attempted=4)
    # >>> doctest.testmod(snapshot)
    # TestResults(failed=0, attempted=4)
    # >>> doctest.testmod(winds)
    # TestResults(failed=0, attempted=4)

    >>> doctest.testmod(galaxy)
    TestResults(failed=0, attempted=18)

'''

from . import utils
from . import config

from .astroconst import *
from . import units
from . import cosmology

from . import simulation
from . import snapshot
from . import winds
from . import galaxy


from . import simlog
from . import derivedtables
from . import progen
from . import accretion

