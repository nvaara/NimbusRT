## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

function(EmbedPTX)
  set(oneArgs OUTPUT_HEADER_FILE INPUT_TARGET)
  set(multiArgs OUTPUT_TARGETS)
  cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

  ## Validate incoming target ##

  get_target_property(INPUT_TARGET_TYPE ${EMBED_PTX_INPUT_TARGET} TYPE)
  if (NOT "${INPUT_TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
    message(FATAL_ERROR "EmbedPTX can only take object libraries")
  endif()

  get_target_property(PTX_PROP ${EMBED_PTX_INPUT_TARGET} CUDA_PTX_COMPILATION)
  if (NOT PTX_PROP)
    message(FATAL_ERROR "'${EMBED_PTX_INPUT_TARGET}' target property 'CUDA_PTX_COMPILATION' must be set to 'ON'")
  endif()

  ## Find bin2c and CMake script to feed it ##

  # We need to wrap bin2c with a script for multiple reasons:
  #   1. bin2c only converts a single file at a time
  #   2. bin2c has only standard out support, so we have to manually redirect to
  #      a cmake buffer
  #   3. We want to pack everything into a single output file, so we need to use
  #      the --name option

  get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
  mark_as_advanced(BIN_TO_C)
  if(NOT BIN_TO_C)
    message(FATAL_ERROR
      "bin2c not found:\n"
      "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
      "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
      )
  endif()

  set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})
  find_file(EMBED_PTX_RUN EmbedPTXRun.cmake)
  mark_as_advanced(EMBED_PTX_RUN)
  if(NOT EMBED_PTX_RUN)
    message(FATAL_ERROR "EmbedPTX.cmake and EmbedPTXRun.cmake must be on CMAKE_MODULE_PATH\n")
  endif()

  ## Create command to run the bin2c via the CMake script ##

  get_filename_component(OUTPUT_FILE_NAME ${EMBED_PTX_OUTPUT_HEADER_FILE} NAME)
  add_custom_command(
    OUTPUT ${EMBED_PTX_OUTPUT_HEADER_FILE}
    COMMAND ${CMAKE_COMMAND}
      "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
      "-DOBJECTS=$<TARGET_OBJECTS:${EMBED_PTX_INPUT_TARGET}>"
      "-DOUTPUT=${EMBED_PTX_OUTPUT_HEADER_FILE}"
      -P ${EMBED_PTX_RUN}
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:${EMBED_PTX_INPUT_TARGET}> ${EMBED_PTX_INPUT_TARGET}
    COMMENT "Generating embedded PTX header file: ${OUTPUT_FILE_NAME}"
  )

  ## Establish dependencies for consuming targets ##

  get_filename_component(OUTPUT_DIR ${EMBED_PTX_OUTPUT_HEADER_FILE} DIRECTORY)
  foreach(OUT_TARGET ${EMBED_PTX_OUTPUT_TARGETS})
    target_include_directories(${OUT_TARGET} PRIVATE ${OUTPUT_DIR})
    target_sources(${OUT_TARGET} PRIVATE ${EMBED_PTX_OUTPUT_HEADER_FILE})
  endforeach()
endfunction()