# Make sure the compiler can find include files from our Hello library. 
include_directories (${ARRAY_SOURCE_DIR}/ARRAY) 

# Make sure the linker can find the Hello library once it is built. 
link_directories (${ARRAY_BINARY_DIR}/ARRAY) 

set (ARRAY_EXAMPLES
performance
)

# iterate over tests
foreach (example ${ARRAY_EXAMPLES})

  # create executable
  add_executable(${example} EXCLUDE_FROM_ALL ${example}.cpp)

  # link executable with the libraries
  target_link_libraries (${example} ${EXTERNAL_LIBS})
#  target_link_libraries (${example} ARRAY ${EXTERNAL_LIBS})

endforeach()


add_custom_target (examples DEPENDS ${ARRAY_EXAMPLES})
