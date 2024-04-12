set_property( GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(glm)
add_subdirectory(pybind11)

set_property(TARGET glm PROPERTY FOLDER "Dependencies")