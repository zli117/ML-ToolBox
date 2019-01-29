#!/usr/bin/env python
import sys
from distutils.core import setup

requirements = [
    'torch>=0.5.0'
]

assert sys.version_info[0] == 3
if sys.version_info[1] < 7:
    requirements.append('dataclasses')

dev_requirements = {
    'dev': [
        'pytype>=2019.1.18',
        'torch',
        'pycodestyle>=2.4.0'
    ]
}

test_requirements = ['pytest>=4.1.1']

setup(name='ML-ToolBox',
      version='0.1',
      description='ML ToolBox for PyTorch',
      author='Zonglin Li',
      author_email='development.my6565@gmail.com',
      url='https://github.com/zli117/ML-ToolBox',
      packages=['toolbox'],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ])