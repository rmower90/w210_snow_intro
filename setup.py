import os
import subprocess
from io import open

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
SEMVER_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

PACKAGE_NAME = "mo_sno"
PROJECT_DESCRIPTION = "A Python package for Snow Water Equivalent (SWE) prediction and data assimilation"
PROJECT_URL = "https://github.com/rmower90/w210_snow_intro/tree/main"

setup(
    name=PACKAGE_NAME,
    version=SEMVER_STRING,
    url=PROJECT_URL,
    description="An API for Snow Water Equivalent (SWE) prediction and data assimilation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "xarray",
        "pandas",
        "geopandas",
        "plotly",
        "streamlit",
        "requests",
        "shapely",
        "cupy",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints"
        ]
    },
    entry_points={
        "console_scripts": [
            "swe-api=app.main:run_api",  
            "swe-dashboard=dashboard.app:main",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
