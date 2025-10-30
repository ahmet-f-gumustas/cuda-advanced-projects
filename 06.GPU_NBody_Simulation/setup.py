"""
GPU N-Body Simulation Python Bindings Setup
Uses pybind11 to expose C++/CUDA code to Python
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON_BINDINGS=ON',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # Add parallel build flag
        build_args += ['--', '-j8']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'nbody_python'] + build_args,
                            cwd=self.build_temp)


# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='nbody-cuda',
    version='1.0.0',
    author='GPU N-Body Project',
    author_email='faruk.gmstss@gmail.com',
    description='GPU-accelerated N-Body gravitational simulation with CUDA',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('nbody_cuda')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'viz': ['matplotlib>=3.3.0'],
        'dev': ['pytest>=6.0.0', 'black>=21.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
    keywords='gpu cuda nbody simulation physics astronomy',
)
