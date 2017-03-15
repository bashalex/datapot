#!/usr/bin/env python

import os
from setuptools import setup, find_packages

CURRENT_DIR = os.path.dirname(__file__)

setup(name='datapot',
      description='Library for automatic feature extraction from JSON-datasets',
      long_description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      version='0.1a1',
      url='https://github.com/bashalex/datapot',
      author='Alex Bash, Yuriy Mokriy, Nikita Savelyev, Michal Rozenwald, Peter Romov',
      author_email='avbashlykov@gmail.com, yurymokriy@gmail.com, n.a.savelyev@gmail.com, michal.rozenwald@gmail.com, romovpa@gmail.com',
      license='Apache2',
      maintainer='Nikita Savelyev',
      maintainer_email='n.a.savelyev@gmail.com',
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'sklearn',
          'iso-639',
          'langdetect',
          'gensim',
          'nltk',
          'tsfresh',
          'python-dateutil',
          'fastnumbers',
          'pystemmer',
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development',
      ],
      packages=find_packages())
