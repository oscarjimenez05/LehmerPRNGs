from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Note: If you are on Windows (MSVC), use '/O2' instead of '-O3' and '-march=native'.
c_args = [
    "-O3",
    "-ffast-math",
    "-funroll-loops",
]

extensions = [
    Extension(
        "c_lcg_lh",
        ["c_lcg_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "xor_lh",
        ["xor_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "raw_xor_lh",
        ["raw_xor_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
]

setup(
    name="Lehmerized Generator Cython Module",
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