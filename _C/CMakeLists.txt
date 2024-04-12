add_subdirectory(VCT)

pybind11_add_module(_C Interface.cpp)
target_link_libraries(_C PRIVATE VCT-Core)
add_custom_command(TARGET _C
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy
                  $<TARGET_FILE:_C>
                  ${CMAKE_SOURCE_DIR}/Python/NimbusRT
                  COMMENT "Copying gs _C module to ${CMAKE_SOURCE_DIR}/Python/NimbusRT")