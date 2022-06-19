import os
import os.path as osp
import shutil
import sys
import warnings

import setuptools
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements(fname='requirements.txt'):
    with open(fname) as f:
        return f.readlines()


if __name__ == '__main__':
    setup(
        name='gitract',
        version='0.0.1',
        description='Hello world',
        packages=setuptools.find_packages(),
        install_requires=parse_requirements('requirements.txt'),
        ext_modules=[],
        zip_safe=False)
