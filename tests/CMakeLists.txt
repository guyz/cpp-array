# Make sure the compiler can find include files from our Hello library. 
include_directories (${cpp-array_SOURCE_DIR}/array)

 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/script.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

set (ARRAY_TESTS test000 test001 test002 test003 test004)

# iterate over tests
foreach (test ${ARRAY_TESTS})

  # create executable
  add_executable(${test} EXCLUDE_FROM_ALL ${test}.cpp)

  # link executable with the libraries
  target_link_libraries (${test} ${EXTERNAL_LIBS})

  # add test
  add_test (${test} ${CMAKE_CURRENT_BINARY_DIR}/script.sh ${CMAKE_CURRENT_BINARY_DIR}/${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.verified)

endforeach()


# create custom target for tests
add_custom_target(
check
${CMAKE_CTEST_COMMAND}
DEPENDS ${ARRAY_TESTS}
)
