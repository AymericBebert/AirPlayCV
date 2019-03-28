#!/usr/bin/env python3

"""
Script to build the Cython code

To build, go to the project root folder and run:
./src/cyvisord/cython-setup.py build_ext --build-lib src/cyvisord
"""

from distutils.core import setup
from distutils.sysconfig import get_config_var
from setuptools import Extension
from Cython.Build import cythonize

import numpy


include_dir = get_config_var('INCLUDEDIR')
include_dirs = [include_dir] if include_dir is not None else []
library_dir = get_config_var('LIBDIR')
library_dirs = [library_dir] if library_dir is not None else []

visord_ext = Extension('visord',
                       ['src/cyvisord/visord.pyx'],
                       include_dirs=include_dirs + [numpy.get_include()],
                       library_dirs=library_dirs,
                       define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")])

setup(ext_modules=cythonize(visord_ext,
                            build_dir='.',
                            annotate=True,
                            compiler_directives={'language_level': 3}))

if __name__ == "__main__":
    pass
