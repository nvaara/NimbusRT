# EmbedPTX

This repo implements a CMake function to help embed PTX using `obj2c`. This
uses the modern CUDA features found in CMake 3.8+ to maximize simplicity of
use.

## Usage

### Incorporating in your project

You can add this code to your CMake build in a number of ways:

- As a git submodule
- Using CMake's [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) module
- Adding the files to your project directly

### Using EmbedPTX()

One you have the scripts available to your CMake build, use it by putting both
`EmbedPTX.cmake` and `EmbedPTXRun.cmake` on `CMAKE_MODULE_PATH` and including
`EmbedPTX`.

Here's a short example, which embeds `myCudaPtx` kernels in a header called
`embedded_ptx.h`:

```cmake
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(example LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH /path/to/EmbedPTX.cmake)
include(EmbedPTX)

add_library(ptx_gen OBJECT myCudaPtx.cu)
set_property(TARGET ptx_gen PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx_gen PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(myCudaExample myCudaExample.cpp)

EmbedPTX(
  OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/embedded_ptx.h
  INPUT_TARGET       ptx_gen
  OUTPUT_TARGETS     myCudaExample
)
```

There are 3 arguments to pass to `EmbedPTX()`:

- `OUTPUT_HEADER_FILE` specifies the header to be generated.
- `INPUT_TARGET` specifies the object library used to generate the PTX files.
- `OUTPUT_TARGETS` is a list of targets which will consume the generated
  header file.

Note that the directory containing `OUTPUT_HEADER_FILE` is automatically added
as a `PRIVATE` include directory to each target passed in `OUTPUT_TARGETS`.
