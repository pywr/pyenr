#!/usr/bin/env python

try:
    from setuptools import setup
    from setuptools import Extension
    print('Using setuptools for setup!')
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    print('Using distutils for setup!')
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import sys
import os
from packaging.version import Version
import subprocess

# with open('README.rst') as fh:
#     long_description = fh.read()

setup_kwargs = {
    'name': 'pywr-dcopf',
    'description': 'Python Water Resource model',
    #'long_description': long_description,
    #'long_description_content_type': 'text/x-rst',
    #'author': 'Joshua Arnott',
    #'author_email': 'josh@snorfalorpagus.net',
    #'url': 'https://github.com/pywr/pywr',
    'packages': ['pywr_dcopf', ],
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'pandas',
        'networkx',
        'scipy',
        'tables',
        'future',
        'xlrd',
        'packaging',
        'matplotlib',
        'jinja2',
        'marshmallow'
    ]
}

define_macros = []

if '--annotate' in sys.argv:
    annotate = True
    sys.argv.remove('--annotate')
else:
    annotate = False

compiler_directives = {}
if '--enable-profiling' in sys.argv:
     compiler_directives['profile'] = True
     sys.argv.remove('--enable-profiling')

build_trace = False
if '--enable-trace' in sys.argv:
    sys.argv.remove('--enable-trace')
    build_trace = True
elif os.environ.get('PYWR_BUILD_TRACE', 'false').lower() == 'true':
    build_trace = True

if build_trace:
    print('Tracing is enabled.')
    compiler_directives['linetrace'] = True
    define_macros.append(('CYTHON_TRACE', '1'))
    define_macros.append(('CYTHON_TRACE_NOGIL', '1'))

compile_time_env = {}
if '--enable-debug' in sys.argv:
    compile_time_env['SOLVER_DEBUG'] = True
    sys.argv.remove('--enable-debug')
else:
    compile_time_env['SOLVER_DEBUG'] = False

# See the following documentation for a description of these directives
#  https://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiler-directives
compiler_directives['language_level'] = 3
compiler_directives['embedsignature'] = True


extensions = [
    Extension('pywr_dcopf._core', ['pywr_dcopf/_core.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr_dcopf._glpk_dcopf_solver', ['pywr_dcopf/_glpk_dcopf_solver.pyx'],
              include_dirs=[np.get_include()],
              libraries=['glpk'],
              define_macros=define_macros),
]

setup_kwargs['package_data'] = {}
# store the current git hash in the module
# try:
#     git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip().decode("utf-8")
# except subprocess.CalledProcessError:
#     pass
# else:
#     with open("pywr_dcopf/GIT_VERSION.txt", "w") as f:
#         f.write(git_hash + "\n")
#     setup_kwargs["package_data"]["pywr_dcopf"] = ["GIT_VERSION.txt"]

# build the core extension(s)
setup_kwargs['ext_modules'] = cythonize(extensions,
                                        compiler_directives=compiler_directives, annotate=annotate,
                                        compile_time_env=compile_time_env)

if os.environ.get('PACKAGE_DATA', 'false').lower() == 'true':
    pkg_data = setup_kwargs["package_data"].get("pywr_dcopf", [])
    pkg_data.extend(['.libs/*', '.libs/licenses/*'])
    setup_kwargs["package_data"]["pywr"] = pkg_data

setup(**setup_kwargs)
