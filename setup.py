from setuptools import find_packages, setup

VERSION = ""
DESCRIPTION = ""

setup(
    name="",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
