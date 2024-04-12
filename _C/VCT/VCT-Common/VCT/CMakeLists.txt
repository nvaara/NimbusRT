set (CXX_SOURCES
    CMakeLists.txt
    Common.hpp
    Constants.hpp
    CudaError.hpp
    DeviceBuffer.cpp
    DeviceBuffer.hpp
    Intersection.hpp
    Kernel.cpp
    Kernel.hpp
    Logger.hpp
    PathStorage.cpp
    PathStorage.hpp
    Profiler.hpp
    Propagation.hpp
    Traversal.hpp
    Types.hpp
    Utils.hpp)

add_library(VCT-Common STATIC ${CXX_SOURCES})
find_package(CUDAToolkit)

SET(OPTIX_8_0_PATH $ENV{OPTIX_8_0_PATH} CACHE FILEPATH "Path to OptiX SDK")

if (NOT OPTIX_8_0_PATH)
    message(FATAL_ERROR "OPTIX_8_0_PATH variable not set in CMake or as a environment variable" )
endif()

target_link_libraries(VCT-Common
  ${PCL_LIBRARIES}
  glm
  CUDA::cuda_driver
  CUDA::cudart_static
 )

target_include_directories(VCT-Common PUBLIC . ${OPTIX_8_0_PATH}/include)
target_compile_definitions(VCT-Common PUBLIC NOMINMAX)