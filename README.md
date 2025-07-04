# NimbusRT 
NimbusRT is a ray launching-based tool for radio channel characterization, which utilizes 3D point clouds as the intersection geometry.

This branch contains the implementation for the paper "Differentiable High-Performance Ray Tracing-Based Simulation of Radio Propagation with Point Clouds"
<details><summary>Previous works</summary>

See branch [vct](https://github.com/nvaara/NimbusRT/tree/vct) for papers  
[Vaara et. al., Ray launching-based computation of exact paths with noisy dense point clouds,” IEEE TAP, 2025](https://ieeexplore.ieee.org/abstract/document/10910023)  
[Vaara et. al., A ray launching approach for computing exact paths with point clouds, ICASSPW, 2024](https://arxiv.org/abs/2402.13747)

</details>

## Reproducing Results
```
1. Download point clouds
2. For measuring the performance of ray tracing, compile Nimbus with uncomment the line #define LOGGING_ENABLED in Profiler.hpp
3. Place the models in the Examples folder. This folder contains the scripts to reproduce results
3. To evaluate performance, run corridor.py with various max_depth in nrt.Params
4. To reproduce the material property results, run material.py
```
## Installing


### Cloning
```shell
git clone https://github.com/nvaara/NimbusRT --recurse-submodules
```

### Prerequisites

<details><summary>Windows</summary>

```
1. Install MSVC C++17 build tools
2. Install CUDA 11.8+
3. Install Optix 8.0
4. Create an environment variable named OPTIX_8_0_PATH that points to the OptiX 8.0 SDK folder
```

</details>

<details><summary>Linux</summary>

#### **gcc-11 g++-11**
```bash
sudo apt install gcc-11 g++-11
```
#### **CUDA Toolkit 12.1**
- First install NVIDIA drivers for your machine, reboot and check if `nvidia-smi` works. 
- Then install `cuda-12.1` using the instructions [here](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)  
- Export CUDA enviroment variables to `~/.bashrc`
```bash
echo '# NVIDIA CUDA' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```
- Reboot, then check installation with `nvcc --version`
#### OptiX 8.0
- Download OptiX SDK from [here](https://developer.nvidia.com/designworks/optix/downloads/legacy). Note you need to have an NVIDIA developer account.
- Execute the shell script and then move the SDK files into a directory of your choice, we call it `<INSTALL_DIR>`.
```bash
# You are most probably in the ~/Downloads directory  
chmod u+x NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
mv NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64 <INSTALL_DIR>
```  
- Generate the cmake build files for the SDK:
```bash
cd <INSTALL_DIR>/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
mkdir build
cd build
ccmake ..
```
- Once in cmake GUI, type in `c`, and type `e` once the process is done. Type `c` again `e` and type `g`. For in-depth explanation see [here](https://www.matthewjmullin.com/posts/optix/). After the generation process is done run `make`
- Check installation and export the environment variables to your `~/.bashrc`. 
```bash
./bin/optixHair  # optionally test sample program, you should see a rendered face with very colorful hairs
echo 'export OPTIX_8_0_PATH=/<absoulte>/<path>/<to>/<INSTALL_DIR>/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64' >> ~/.bashrc
```
</details>


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

### Examples

Coming soon!
