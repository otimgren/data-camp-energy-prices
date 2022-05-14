from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "A data science project"

setup(
    name="energy_prices",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
