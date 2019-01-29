import sys
from subprocess import Popen

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()


def install(packages):
    for package in packages:
        process = Popen(['pip', 'install', '--progress-bar', 'off', package])
        print(process.args)
        process.wait()


if __name__ == '__main__':
    install(requirements)
    assert sys.version_info[0] == 3
    if sys.version_info[1] < 7:
        install(['dataclasses'])
