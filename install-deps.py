import os
import sys
from subprocess import Popen

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()


def install(packages):
    with open(os.devnull, 'w') as fp:
        for package in packages:
            process = Popen(['pip', 'install', package], stdout=fp)
            process.wait()


if __name__ == '__main__':
    install(requirements)
    assert sys.version_info[0] == 3
    if sys.version_info[1] < 7:
        install(['dataclasses'])
