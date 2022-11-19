from setuptools import find_packages, setup

setup(
    name="drcomp",
    version="0.0.0",
    author="Moritz Mistol",
    author_email="moritz.mistol@gmail.com",
    description="Python package for the comparison of dimensionality reduction techniques",
    packages=find_packages(include=["drcomp", "drcomp.*"]),
)
