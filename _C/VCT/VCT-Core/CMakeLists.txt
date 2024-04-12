set(CXX_SOURCES 
                VoxelConeTracer.cpp
                VoxelConeTracer.hpp
                KernelData.cpp
                KernelData.hpp
                InputData.hpp)

add_library(VCT-Core STATIC ${CXX_SOURCES})
target_link_libraries(VCT-Core PUBLIC VCT-Common)
target_include_directories(VCT-Core PUBLIC .)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/Dependencies/EmbedPTX)
include(EmbedPTX)

EmbedPTX(
  OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/PtxData.cpp
  INPUT_TARGET       VCT-Ptx
  OUTPUT_TARGETS     VCT-Core
)