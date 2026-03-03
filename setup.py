#!/usr/bin/env python
"""Setup script for sop_lake package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sop_lake",
    version="0.1.0",
    author="Alessandro Carbone",
    author_email="alessandro.carbone@epfl.ch",
    description="Dynamical Mean Field Theory from a sum-over-pole perspective",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlessandroCarbone/sop_lake",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pyyaml>=5.3",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
)
