from setuptools import setup, find_packages

def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(name="nimbusrt", version="1.0.0", packages=find_packages(), install_requires=parse_requirements("requirements.txt"),)
