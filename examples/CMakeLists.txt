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

# Make sure the linker can find the Hello library once it is built. 
link_directories (${cpp-array_BINARY_DIR}/array)

if (CUDA)

  set (ARRAY_EXAMPLES
    cuda_gemm_performance
    cuda_gemv_performance
    cuda_dot_performance
  )
  
else()

  set (ARRAY_EXAMPLES
    gemm_performance
    gemv_performance
    dot_performance
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
