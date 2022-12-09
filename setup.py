from setuptools import find_packages, setup

setup(
    name="drcomp",
    version="0.1.0",
    author="Moritz Mistol",
    author_email="moritz.mistol@gmail.com",
    description="Python package for the comparison of dimensionality reduction techniques",
    packages=find_packages(include=["drcomp", "drcomp.*"]),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["drcomp=drcomp.__main__:main"]},
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "torchvision",
        "scikit-dimension",
        "skorch",
        "hydra-core",
    ],
)
