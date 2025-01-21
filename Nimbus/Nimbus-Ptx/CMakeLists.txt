set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/HitGroupCommon.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Ray.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/PathRefiner.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/SDF.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Primitive.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Trace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Triangle.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Visibility.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Refine.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Ptx_Coverage.cu
    )

set(CUH_SOURCES "")
foreach(file ${CUDA_SOURCES})
    get_filename_component(extension ${file} EXT)
    if(extension STREQUAL ".cuh")
        list(APPEND CUH_SOURCES ${file})
    endif()
endforeach()
source_group("CUDA Header Files" FILES ${CUH_SOURCES})

add_library(Nimbus-Ptx OBJECT ${CUDA_SOURCES})
target_link_libraries(Nimbus-Ptx PUBLIC Nimbus-Common)
set_target_properties(Nimbus-Ptx PROPERTIES CUDA_PTX_COMPILATION ON)
set_target_properties(Nimbus-Ptx PROPERTIES CUDA_ARCHITECTURES OFF)

target_compile_options(Nimbus-Ptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math
                        --diag-suppress 20012>
                        )