from setuptools import setup, find_packages

setup(
    name="dlabrenishaw",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "requests>=2.26.0",
        "pycromanager>=0.25.0",
    ],
    python_requires=">=3.12",
    author="Antony Georgiadis",
    author_email="antonyg@stanford.edu",
    description="Modular control system for Raman-fluorescence microscopy",
    url="https://github.com/a-georgiadis/DlabRenishaw",
)