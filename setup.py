#!/usr/bin/env python
import sys
from distutils.core import setup

requirements = [
    'torch'
]

assert sys.version_info[0] == 3
if sys.version_info[1] < 7:
    requirements.append('dataclasses')

dev_requirements = {
    'dev': [
        'mypy>=0.660',
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
      install_requires=requirements,
      extras_require=dev_requirements,
      tests_require=test_requirements,
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ])
