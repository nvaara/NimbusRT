# NimbusRT 

This branch contains experimental support for Ubuntu 20.04.

## Installing

### Prerequisites

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