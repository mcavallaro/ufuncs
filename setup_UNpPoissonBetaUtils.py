"""
Usage: CFLAG=-03 python2 setup_UNpPoissonBetaUtils.py build_ext --inplace --optimize
"""
from distutils.core import setup, Extension

# PoissonBetaUtils = Extension('UNpPoissonBetaUtils',
#                    libraries = ['gsl', 'gslcblas', 'm'],
#                    sources = ['UNpPoissonBetaUtils.c'])

# setup (name = 'PoissonBetaUtils',
#        version = '1.0',
#        description = 'This package contains C implementations utils for the PoissonBeta model',
#        ext_modules = [PoissonBetaUtils])


UNpPoissonBetaUtils = Extension('UNpPoissonBetaUtils',
                    libraries = ['gsl', 'gslcblas', 'm'],
                    language = ['c'],
                    sources = ['UNpPoissonBetaUtils.c'])

setup (name = 'UNpPoissonBetaUtils',
       version = '1.0',
       description = 'This package contains C-implementations utils for the PoissonBeta model with NumPy support',
       ext_modules = [UNpPoissonBetaUtils])