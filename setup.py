from setuptools import setup, find_packages

setup(
    name="sterics_descriptors",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],  # dependencies are in pyproject.toml
    author="Alessio Prunotto",
    description="Sterics descriptors computation tools",
    url="https://github.com/AlessioPrunotto/sterics-descriptors",
)
