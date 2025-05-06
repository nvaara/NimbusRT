# NimbusRT

NimbusRT is a ray launching-based tool for radio channel characterization, which utilizes point clouds as the intersection geometry. It computes paths consisting of specular reflections and diffractions between a set of transmitters and receivers.
This branch contains the code for papers:  
[Ray Launching-Based Computation of Exact Paths with Noisy Dense Point Clouds
](https://ieeexplore.ieee.org/abstract/document/10910023)  
[A Ray Launching Approach for Computing Exact Paths with Point Clouds](https://arxiv.org/abs/2402.13747)

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
cmake ..
cmake --build . --config Release
```

### Python Installation

In the root directory:
```shell
pip install -e Python
```

## Running

See Examples folder for demo scripts. The point clouds can be downloaded [here](https://drive.google.com/drive/folders/1X8U4hZziVi5zpp93a1eYDvXLJL8103ZY).

## Citation
Journal paper:
```bibtex
@article{vaara2025ray,
  title={Ray Launching-Based Computation of Exact Paths with Noisy Dense Point Clouds},
  author={Vaara, Niklas and Sangi, Pekka and L{\'o}pez, Miguel Bordallo and Heikkil{\"a}, Janne},
  journal={IEEE Transactions on Antennas and Propagation},
  year={2025},
  publisher={IEEE}
}
```
Conference paper:
```bibtex
@inproceedings{vaara2024ray,
  author={Vaara, Niklas and Sangi, Pekka and López, Miguel Bordallo and Heikkilä, Janne},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={A Ray Launching Approach for Computing Exact Paths with Point Clouds}, 
  year={2024},
  pages={800-804},
  doi={10.1109/ICASSPW62465.2024.10626030}}
```