from setuptools import find_packages
from setuptools import setup

setup(
    name='openvio_slam',
    version='1.0.0',
    packages=find_packages(
        include=('openvio_slam', 'openvio_slam.*')),
)
