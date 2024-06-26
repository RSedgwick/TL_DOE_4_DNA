from setuptools import find_packages, setup
import pathlib as pl

VERSION = '0.1.0'
DISTNAME = "candas"
DESCRIPTION = "Competitive Amplification Network Design and Analysis Suite"
AUTHOR = "John Goertz"
AUTHOR_EMAIL = ""
URL = ""
LICENSE = ""

PROJECT_ROOT = pl.Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

with open(REQUIREMENTS) as f:
    install_reqs = f.read().splitlines()

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]

setup(
    # the name must match the folder name 'verysimplemodule'
    name=DISTNAME,
    version=VERSION,
    author="John Goertz",
    author_email="",
    description=DESCRIPTION,
    #long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=install_reqs,
    classifiers=classifiers,
    #keywords=['python'],
)