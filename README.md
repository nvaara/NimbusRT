# NimbusRT

NimbusRT is a ray launching-based tool for radio channel characterization, which utilizes point clouds as the intersection geometry. It computes paths consisting of specular reflections and diffractions between a set of transmitters and receivers.

[ArXiv] [Ray Launching-Based Computation of Exact Paths with Noisy Dense Point Clouds
](https://arxiv.org/abs/2403.06648)

[ArXiv] [A Ray Launching Approach for Computing Exact Paths with Point Clouds](https://arxiv.org/abs/2402.13747)

## Installing


NimbusRT has been tested with CUDA 11.8+, OptiX 8.0 and Python 3.9+.
So far, it has only been tested on Windows 11.

### Prerequisites
```
1. Install MSVC C++17 build tools
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

### Compiling
In the root directory of NimbusRT:
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Python Installation

In the root directory:
```shell
cd Python
pip install -e .
```

## Running

See Examples folder for demo scripts.

## Citation
Journal paper:
```bibtex
@article{vaara2024ray,
  title={Ray Launching-Based Computation of Exact Paths with Noisy Dense Point Clouds},
  author={Vaara, Niklas and Sangi, Pekka and L{\'o}pez, Miguel Bordallo and Heikkil{\"a}, Janne},
  journal={arXiv preprint arXiv:2403.06648},
  year={2024}
}
```
Conference paper:
```
@article{vaara2024ray,
  title={A Ray Launching Approach for Computing Exact Paths with Point Clouds},
  author={Vaara, Niklas and Sangi, Pekka and L{\'o}pez, Miguel Bordallo and Heikkil{\"a}, Janne},
  journal={arXiv preprint arXiv:2402.13747},
  year={2024}
}
```