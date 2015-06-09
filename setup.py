from distutils.core import setup, Extension
import numpy

extension = Extension('_psf', ['spindle_tracker/movies/psf.c'],
                      include_dirs=[numpy.get_include()])

setup(name='_psf', ext_modules=[extension])
