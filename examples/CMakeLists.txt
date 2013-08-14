# Make sure the compiler can find include files from our Hello library. 
include_directories (${cpp-array_SOURCE_DIR}/array)

# Make sure the linker can find the Hello library once it is built. 
link_directories (${cpp-array_BINARY_DIR}/array)

if (CUDA)

  set (ARRAY_EXAMPLES
    cuda_performance
  )

else()

  set (ARRAY_EXAMPLES
    performance
  )

endif()


# iterate over tests
foreach (example ${ARRAY_EXAMPLES})

  # create executable
  add_executable(${example} EXCLUDE_FROM_ALL ${example}.cpp)

  # link executable with the libraries
  target_link_libraries (${example} ${EXTERNAL_LIBS})

endforeach()


add_custom_target (examples DEPENDS ${ARRAY_EXAMPLES})
