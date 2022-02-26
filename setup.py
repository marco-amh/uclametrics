# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:08:18 2021

@author: marco
"""

from setuptools import setup

setup(
   name='SuperImporter',
   version='0.1.0',
   author='Marco Martinez Huerta',
   author_email='marco.martinez@ucla.edu',
   packages=['SuperImporter'],
   url='http://pypi.python.org/pypi/PackageName/',
   license='MIT',
   description='Pull data from many sites like Bank of Mexico, Inegi, Fred, IMF, WB, Google Trends',
   #long_description=open('README.txt').read(),
   install_requires=[
       "Django >= 1.1.1",
       "pytest",
   ],
)
