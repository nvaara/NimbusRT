# NimbusRT - Under construction

This branch contains experimental things for now.

## Installing


NimbusRT has been tested with CUDA 11.8+, OptiX 8.0 and Python 3.9+.
So far, it has only been tested on Windows 11.

### Prerequisites
```
1. Install MSVC C++17 build tools (Windows)
2. Install CUDA 11.8+
3. Install Optix 8.0
4. Create an environment variable named OPTIX_8_0_PATH that points to the OptiX 8.0 SDK folder
```

### Cloning
```shell
git clone https://github.com/nvaara/NimbusRT --recurse-submodules
```

### Python Environment

We suggest using anaconda:
```shell
conda create --name NimbusRT python=3.10
conda activate NimbusRT
```

### Python Installation

In the root directory:

```shell
pip install .
```
or for editable mode:
```shell
pip install -e .
```