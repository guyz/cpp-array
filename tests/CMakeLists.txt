################################################################################
#
# Copyright (©) 2014 Alejandro M. Aragón
# Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
# All Rights Reserved
#
# cpp-array is free  software: you can redistribute it and/or  modify it under
# the terms  of the  GNU Lesser  General Public  License as  published by  the 
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# cpp-array is  distributed in the  hope that it  will be useful, but  WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A  PARTICULAR PURPOSE. See  the GNU  Lesser General  Public License  for 
# more details.
#
# You should  have received  a copy  of the GNU  Lesser General  Public License
# along with cpp-array. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################


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
