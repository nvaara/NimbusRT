from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

def parse_subprocess_result(res, error_msg):
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        sys.exit(error_msg, res.stderr)

def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

if os.getenv("OPTIX_8_0_PATH") is None:
    sys.exit("Please set environment variable OPTIX_8_0_PATH to point to OptiX 8.0 installation directory.")

build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
print("BUILD_DIR: ", build_dir)

if not os.path.exists(build_dir):
    os.makedirs(build_dir)

res_gen = subprocess.run(["cmake", ".."], cwd=build_dir, capture_output=True, text=True)        
parse_subprocess_result(res_gen, "Failed to generate CMake project.")
res_build = subprocess.run(['cmake', '--build', '.', '--config', 'Release'], cwd=build_dir, capture_output=True, text=True)
parse_subprocess_result(res_build, "Failed to build CMake project.")

setup(name="nimbusrt",
      version="1.0.0",
      packages=find_packages(),
      install_requires=parse_requirements("requirements.txt"),
      include_package_data=True,
      package_data={'_C': ['*.pyd']},
      zip_safe=False)