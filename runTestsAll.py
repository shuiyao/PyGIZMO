#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all doctests for the PyGIZMO module.
"""

import pygizmo
import doctest
import sys
import warnings

warnings.filterwarnings("ignore")
# print("*********************************************************************")
# print("pygizmo version ", pygizmo.version)
# print("*********************************************************************")
# pygad.environment.verbose = pygad.environment.VERBOSE_NORMAL
print("running pygizmo doctest...")

res = doctest.testmod(pygizmo)

print("*********************************************************************")
print("return code = ", res.failed)

# sys.exit(res.failed)
