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

# - Find CLAPACK
# Find the native CLAPACK headers and libraries.
#
#  CLAPACK_FOUND        - True if clapack found
#  CLAPACK_INCLUDE_DIR  - Directory of clapack header files
#  CLAPACK_LIBRARY      - List of libraries when using clapack
#

message (STATUS "Looking for clapack library")

# find header file
find_path (CLAPACK_INCLUDE_DIR NAMES clapack.h PATH_SUFFIXES atlas)

if (CLAPACK_INCLUDE_DIR)

  message(STATUS "CLAPACK include directory ${CLAPACK_INCLUDE_DIR}")
  set (CLAPACK_HEADER ${CLAPACK_INCLUDE_DIR}/clapack.h)

  # chdck for Apple implementation                                                                                                             
  string(FIND ${CLAPACK_INCLUDE_DIR} "vecLib.framework" APPLE_MATCHED)

  if (NOT APPLE_MATCHED MATCHES -1)
    set(CLAPACK_APPLE 1)
    message(STATUS "Found Apple clapack implementation, setting CLAPACK_APPLE to ${CLAPACK_APPLE}")
  else()

    # check for atlas implementation                                                                                                           
    string(FIND ${CLAPACK_INCLUDE_DIR} "atlas" ATLAS_MATCHED)

    if (NOT ATLAS_MATCHED MATCHES -1)
      set(CLAPACK_ATLAS 1)
      message(STATUS "Found Atlas clapack implementation, setting CLAPACK_ATLAS to ${CLAPACK_ATLAS}")
    endif()
  endif()
endif()


# find library
find_library (CLAPACK_LIBRARY NAMES lapack lapack_atlas PATHS ${USER_LIB_PATH}/lib ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_SOURCE_DIR}/../lib PATH_SUFFIXES atlas atlas-base)

set(CLAPACK_LIBRARIES ${CLAPACK_LIBRARY})
set(CLAPACK_INCLUDE_DIRS ${CLAPACK_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLAPACK DEFAULT_MSG
                                  CLAPACK_INCLUDE_DIR CLAPACK_LIBRARY)

mark_as_advanced(CLAPACK_INCLUDE_DIR CLAPACK_LIBRARY)