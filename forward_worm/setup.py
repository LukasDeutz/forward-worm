'''
Created on 28 Jun 2022

@author: lukas
'''
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'forward_worm',
    version = '0.1',
    author = 'Lukas Deutz',
    author_email = 'scld@leeds.ac.uk',
    description = 'Module simple-worm experiment parameter sweeps',
    url = 'https://github.com/LukasDeutz/forward-worm.git',
    packages = find_packages()
)


