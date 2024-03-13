# setup.py is responsible in creating ML application as a package

from setuptools import find_packages,setup
from typing import List

E_DOT = "-e ."
def get_requirements(filepath)->List[str]:
    requirements = []
    with open(filepath) as inst:
        requirements = inst.readlines()
        requirements = [req.replace('\n','') for req in requirements]

    if E_DOT in requirements:
        requirements.remove(E_DOT)

    
setup(
    name = "student_performance_indicator",
    version = "0.0.1",
    author = "umair",
    author_email = "umairsiddique3171@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)