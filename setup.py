from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="deep-skull",
    version="0.1.0",
    description="A CT head skull stripping command line tool, using the model and weights from CT_BET: https://github.com/aqqush/CT_BET",
    url="https://github.com/jdddog/deep-skull",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={"console_scripts": ["deep-skull = deep_skull.cli:cli"]},
)
