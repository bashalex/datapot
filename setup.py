#!/usr/bin/env python

import os
from setuptools import setup, find_packages

CURRENT_DIR = os.path.dirname(__file__)

setup(name='datapot',
      description='Library for automatic feature extraction from JSON-datasets',
      long_description=open(os.path.join(CURRENT_DIR, 'README.rst')).read(),
      version='0.1.2',
      url='https://github.com/bashalex/datapot',
      author='Alex Bash, Yuriy Mokriy, Nikita Saveyev, Michal Rozenwald, Peter Romov',
      author_email='avbashlykov@gmail.com, yurymokriy@gmail.com, n.a.savelyev@gmail.com, michal.rozenwald@gmail.com, romovpa@gmail.com',
      license='GNU v3.0',
      maintainer='Nikita Savelyev',
      maintainer_email='n.a.savelyev@gmail.com',
      install_requires=[
          'numpy >= 1.6.1',
          'scipy >= 0.17.0',
          'pandas >= 0.17.1',
          'scikit-learn >= 0.17.1',
          'iso-639 >= 0.4.5',
          'langdetect >= 1.0.7',
          'gensim >= 2.1.0',
          'nltk >= 3.2.4',
          'tsfresh >= 0.7.1',
          'python-dateutil >= 2.6.0',
          'fastnumbers >= 2.0.1',
          'pystemmer >= 1.3.0',
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development',
      ],
      packages=find_packages())
