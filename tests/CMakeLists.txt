# Make sure the compiler can find include files from our Hello library. 
include_directories (${ARRAY_SOURCE_DIR}/ARRAY) 

set (ARRAY_TESTS test001)

# iterate over tests
foreach (test ${ARRAY_TESTS})

  # create executable
  add_executable(${test} EXCLUDE_FROM_ALL ${test}.cpp)

  # link executable with the libraries
  target_link_libraries (${test} ${CBLAS_LIB})
#  target_link_libraries (${test} ARRAY ${CBLAS_LIB})

  # add test
  add_test (${test} ${CMAKE_CURRENT_SOURCE_DIR}/script.sh ${CMAKE_CURRENT_BINARY_DIR}/${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.verified)

endforeach()


# create custom target for tests
add_custom_target(
check
${CMAKE_CTEST_COMMAND}
DEPENDS ${ARRAY_TESTS}
)
