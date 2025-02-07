set(CXX_SOURCES 
                Nimbus/CudaError.hpp
                Nimbus/CudaUtils.cpp
                Nimbus/CudaUtils.hpp
                Nimbus/DeviceBuffer.cpp
                Nimbus/DeviceBuffer.hpp
                Nimbus/Environment.cpp
                Nimbus/Environment.hpp
                Nimbus/Hash.hpp
                Nimbus/KernelData.cpp
                Nimbus/KernelData.hpp
                Nimbus/Logger.hpp
                Nimbus/PathStorage.cpp
                Nimbus/PathStorage.hpp
                Nimbus/PointCloudEnvironment.cpp
                Nimbus/PointCloudEnvironment.hpp
                Nimbus/Profiler.hpp
                Nimbus/ScatterTracer.cpp
                Nimbus/ScatterTracer.hpp
                Nimbus/TriangleMeshEnvironment.cpp
                Nimbus/TriangleMeshEnvironment.hpp
                )

add_library(Nimbus STATIC ${CXX_SOURCES})
target_link_libraries(Nimbus PUBLIC Nimbus-Common)
set_property(TARGET Nimbus PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(Nimbus PUBLIC Nimbus-Common stdc++ -static)
target_include_directories(Nimbus PUBLIC .)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/Dependencies/EmbedPTX)
include(EmbedPTX)

EmbedPTX(
  OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/PtxData.cpp
  INPUT_TARGET       Nimbus-Ptx
  OUTPUT_TARGETS     Nimbus
)