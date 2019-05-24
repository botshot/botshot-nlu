# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

# with open("botshot/version.py") as fp:
#     version_dict = {}
#     exec(fp.read(), version_dict)
#     __version__ = version_dict['__version__']

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fp:
    requirements = fp.read().split("\n")


setup(
    name='botshot-nlu',
    version="0.0.1-alpha",
    packages=find_packages(),
    include_package_data=True,
    description='A tiny NLU wrapper/library for Botshot.',
    long_description=README,
    url='https://github.com/botshot/botshot-nlu',
    author='Matúš Žilinec',
    author_email='zilinec.m@gmail.com',
    entry_points={
          'console_scripts': [
              'bots-nlu = botshot_nlu.cli:main'
          ]
      },
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    install_requires=requirements,
)
