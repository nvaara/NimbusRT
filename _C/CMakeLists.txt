set(CXX_SOURCES 
                CMakeLists.txt
                ModuleEntry.cpp
                Scene.cpp
                Scene.hpp
                Wrappers.cpp
                Wrappers.hpp
    )

pybind11_add_module(_C ${CXX_SOURCES})
target_link_libraries(_C PRIVATE Nimbus)
add_custom_command(TARGET _C
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy
                  $<TARGET_FILE:_C>
                  ${CMAKE_SOURCE_DIR}/nimbusrt
                  COMMENT "Copying _C module to ${CMAKE_SOURCE_DIR}/nimbusrt")