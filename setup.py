#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path
from setuptools import setup, find_packages

CURRENT_PATH = path.abspath(path.dirname(__file__))

with open(path.join(CURRENT_PATH, "README.md")) as readme_file:
    readme = readme_file.read()

with open(path.join(CURRENT_PATH, "CHANGELOG.md")) as changelog_file:
    changelog = changelog_file.read()

requirements = ["numpy", "matplotlib", "scipy", "gdal", "shapely>=2", "pyproj", "scikit-image", "scikit-learn"]

setup(
    name="s2lx",
    version="0.1.0",
    description="Simple Sentinel-2 tools",
    long_description=readme + "\n\n" + changelog,
    long_description_content_type="text/markdown",
    author="Ondrej Lexa",
    author_email="lexa.ondrej@gmail.com",
    url="https://github.com/ondrolexa/s2lx",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    zip_safe=False,
    keywords="s2lx",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
    ],
)
