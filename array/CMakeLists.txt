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


# find all source files to build the library
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ARRAY_LIB_SRC)
list(REMOVE_ITEM ARRAY_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# add library array
#add_library (array ${ARRAY_LIB_SRC})

# link library with other libraries
#target_link_libraries (array ${EXTERNAL_LIBS})

# specify installation of the library and the header files
#install (TARGETS array DESTINATION lib)

# header files to install
file(GLOB HEADER_FILES ${PROJECT_CURRENT_SOURCE_DIR} "*.hpp")
list(APPEND HEADER_FILES ${PROJECT_BINARY_DIR}/array-config.hpp)
install (FILES  ${HEADER_FILES} DESTINATION include/array)

# also install cuda directory
if (CUDA)
  install (DIRECTORY ${PROJECT_CURRENT_SOURCE_DIR}cuda DESTINATION include/array)
endif ()



# dev option for development
option(CPP-ARRAY_DEV "Compile main.cpp within array directory for development" OFF)

if (CPP-ARRAY_DEV)

  # add executable
  add_executable(dev.out main.cpp)

  # link executable
  target_link_libraries (dev.out ${EXTERNAL_LIBS})

endif()