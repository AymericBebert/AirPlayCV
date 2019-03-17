#!/usr/bin/env python3

"""
Script to build the Cython code

To build, go to the project root folder and run:
./src/cyvisord/cython-setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('src/cyvisord/visord.pyx',
                            build_dir='.',
                            annotate=True,
                            compiler_directives={'language_level': 3}),
      include_dirs=[numpy.get_include()])

# annotate=True enables generation of the html annotation file

if __name__ == "__main__":
    pass
