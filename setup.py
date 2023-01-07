from setuptools import find_packages, setup

setup(
    name="drcomp",
    version="0.1.0",
    author="Moritz Mistol",
    author_email="moritz.mistol@gmail.com",
    description="Python package for the comparison of dimensionality reduction techniques",
    packages=find_packages(include=["drcomp", "drcomp.*"]),
    python_requires=">=3.9",
    include_package_data=True,
    entry_points={"console_scripts": ["drcomp=drcomp.__main__:main"]},
    install_requires=[
        "matplotlib>=3.6.2",
        "numpy>=1.23.4",
        "pandas>=1.5.1",
        "scikit-learn>=1.2.0",
        "scikit-dimension>=0.3.2",
        "torch>=1.13.0",
        "skorch>=0.12.1",
        "torchvision>=0.14.0",
        "torchinfo>=1.7.1",
        "hydra-core>=1.3.0",
        "SciencePlots>=2.0.1",
        "numba>=0.56.4",
        "joblib>=1.2.0",
        "tqdm>=4.64.1",
    ],
)
