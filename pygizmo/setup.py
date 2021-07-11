#!/usr/bin/env python
import os
import subprocess
import sys
from glob import glob

from setuptools import Extension, setup

# import versioneer

# define scripts
scripts = []

# find all sub-packages
# __file__ = "/home/shuiyao/codes/pygizmo/setup.py"
__file__ = "/home/shuiyao_umass_edu/pygizmo/setup.py"

modules = []
setup_dir = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(setup_dir):
    submod = os.path.relpath(root, setup_dir).replace(os.sep, ".")
    if not submod.startswith("pygad"):
        continue
    if "__init__.py" in files:
        modules.append(submod)

# clean and make the cpygizmo.so library
subprocess.run(["make", "clean"], cwd=setup_dir + "/C", check=True)
ext_module = Extension(
    "C/cpygizmo",
    language="gcc",
    sources=glob("C/src/*"),
    include_dirs=["C/include", "/usr/include"],
    extra_compile_args=[
        "-fPIC",
        "-O3",
        "-Wall",
        "-Wextra",        
    ],
    libraries=["m", "z", "hdf5"],
)

setup(
    name="cpygizmo",
    description="analysis module for GIZMO",
    author="Shuiyao Huang",
    author_email="shuangumass@gmail.com",
    url="",
    include_package_data=True,
    packages=list(map(str, modules)),
    scripts=scripts,
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    ext_modules=[ext_module],
)
