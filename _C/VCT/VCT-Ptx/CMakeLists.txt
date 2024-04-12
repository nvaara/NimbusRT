set(CUDA_SOURCES

    ${CMAKE_CURRENT_SOURCE_DIR}/HitGroupCommon.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Interaction.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Propagation.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Ray.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/PathRefiner.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/SDF.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureTraverser.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_VCT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Voxelization.cu
    )

set(CUH_SOURCES "")
foreach(file ${CUDA_SOURCES})
    get_filename_component(extension ${file} EXT)
    if(extension STREQUAL ".cuh")
        list(APPEND CUH_SOURCES ${file})
    endif()
endforeach()
source_group("CUDA Header Files" FILES ${CUH_SOURCES})

add_library(VCT-Ptx OBJECT ${CUDA_SOURCES})
target_link_libraries(VCT-Ptx PUBLIC VCT-Common)
set_target_properties(VCT-Ptx PROPERTIES CUDA_PTX_COMPILATION ON)
set_target_properties(VCT-Ptx PROPERTIES CUDA_ARCHITECTURES OFF)

target_compile_options(VCT-Ptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math
                        --diag-suppress 20012>
                        )