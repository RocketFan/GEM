from distutils.core import setup
from setuptools import find_packages

setup(
    name='gem',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)
