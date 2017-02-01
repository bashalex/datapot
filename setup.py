#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='datapot',
      version='0.1',
      description='Datapot Python library',
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
      ],
      url='https://github.com/bashalex/datapot',
      packages=find_packages())

