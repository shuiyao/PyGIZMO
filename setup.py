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

modules = []
setup_dir = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(setup_dir):
    submod = os.path.relpath(root, setup_dir).replace(os.sep, ".")
    if not submod.startswith("pygizmo"):
        continue
    if "__init__.py" in files:
        modules.append(submod)

raise ValueError        

# clean and make the cpygizmo.so library
subprocess.run(["make", "clean"], cwd=setup_dir + "/pygizmo/C", check=True)
ext_module = Extension(
    "pygizmo/C/cpygizmo",
    language="gcc",
    sources=glob("pygizmo/C/src/*"),
    include_dirs=["pygizmo/C/include", "/usr/include"],
    extra_compile_args=[
        "-fPIC",
        "-O3",
        "-Wall",
        "-Wextra",        
    ],
    libraries=["m", "z", "hdf5"],
)

setup(
    name="pygizmo",
    description="Python APIs for GIZMO simulations",
    author="Shuiyao Huang",
    author_email="shuangumass@gmail.com",
    url="https://github.com/shuiyao/PyGIZMO",
    include_package_data=True,
    packages=list(map(str, modules)),
    package_data={'pygizmo': ['pygizmo.cfg', 'data/*']},
    scripts=scripts,
    version='1.0',
    ext_modules=[ext_module],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only'
    ],
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
)
