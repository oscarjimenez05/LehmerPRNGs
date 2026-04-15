from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

c_args = ["-O3", "-ffast-math", "-funroll-loops"]

extensions = [
    Extension(
        "alternatives.lcg_fenwick",
        ["alternatives/lcg_fenwick.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "alternatives.xor_fenwick",
        ["alternatives/xor_fenwick.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "alternatives.logistic_lh",
        ["alternatives/logistic_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "alternatives.gaussian_lh",
        ["alternatives/gaussian_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
    ),
    Extension(
        "alternatives.slope_lh",
        ["alternatives/slope_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
    ),
    Extension(
        "alternatives.decay_lh",
        ["alternatives/decay_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
    ),
    Extension(
        "alternatives.comp",
        ["alternatives/comp.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
    ),
]

setup(
    name="Lehmerized Generator Alternatives",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'initializedcheck': False,
            'language_level': "3",
        }
    ),
)