# - Find CBLAS
# Find the native CBLAS headers and libraries.
#
#  CBLAS_LIBRARIES    - List of libraries when using cblas.
#  CBLAS_FOUND        - True if cblas found.
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
find_library (CBLAS_LIBRARY NAMES cblas gslcblas PATHS ${USER_LIB_PATH}/lib $ENV{SUPERLUDIR}/lib ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_SOURCE_DIR}/../lib)

set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
set(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG
                                  CBLAS_INCLUDE_DIR CBLAS_LIBRARY)

mark_as_advanced(CBLAS_INCLUDE_DIR CBLAS_LIBRARY)
