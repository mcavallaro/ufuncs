"""
Usage: python2 setup_toy.py build_ext --inplace
"""

def configuration(parent_package='', top_path=None):
  import numpy
  from numpy.distutils.misc_util import Configuration

  config = Configuration('', parent_package, top_path)

  config.add_extension('nmf', ['setup_toy.c'])

  return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
