set (CXX_SOURCES
                CMakeLists.txt
                Optix.cpp
                Nimbus/Constants.hpp
                Nimbus/Types.hpp
                Nimbus/Utils.hpp)

add_library(Nimbus-Common STATIC ${CXX_SOURCES})
find_package(CUDAToolkit)

SET(OPTIX_8_0_PATH $ENV{OPTIX_8_0_PATH} CACHE FILEPATH "Path to OptiX SDK")

if (NOT OPTIX_8_0_PATH)
    message(FATAL_ERROR "OPTIX_8_0_PATH variable not set in CMake or as a environment variable" )
endif()

target_link_libraries(Nimbus-Common
                                glm
                                CUDA::cuda_driver
                                CUDA::cudart_static
 )

target_include_directories(Nimbus-Common PUBLIC . ${OPTIX_8_0_PATH}/include)
target_compile_definitions(Nimbus-Common PUBLIC NOMINMAX)