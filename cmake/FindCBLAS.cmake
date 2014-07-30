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

# - Find CBLAS
# Find the native CBLAS headers and libraries.
#
#  CBLAS_FOUND        - True if cblas found.
#  CBLAS_INCLUDE_DIR  - Directory of cblas header files
#  CBLAS_LIBRARIES    - List of libraries when using cblas.
#

message (STATUS "Looking for cblas library")

# find header file
find_path (CBLAS_INCLUDE_DIR NAMES cblas.h gsl_cblas.h
PATH_SUFFIXES gsl
)

if (CBLAS_INCLUDE_DIR)  
  if (EXISTS ${CBLAS_INCLUDE_DIR}/cblas.h)
    set (CBLAS_HEADER ${CBLAS_INCLUDE_DIR}/cblas.h)
  elseif (EXISTS ${CBLAS_INCLUDE_DIR}/gsl_cblas.h)
    set (CBLAS_HEADER ${CBLAS_INCLUDE_DIR}/gsl_cblas.h)
  endif()
endif(CBLAS_INCLUDE_DIR)

# find library
find_library (CBLAS_LIBRARY NAMES cblas gslcblas PATHS ${USER_LIB_PATH}/lib ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_SOURCE_DIR}/../lib)

set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
set(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG
                                  CBLAS_INCLUDE_DIR CBLAS_LIBRARY)

mark_as_advanced(CBLAS_INCLUDE_DIR CBLAS_LIBRARY)
