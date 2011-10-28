# The name of our project is ARRAY. CMakeLists files in this project can 
# refer to the root source directory of the project as ${ARRAY_SOURCE_DIR}
# and to the root binary directory of the project as ${ARRAY_BINARY_DIR}

cmake_minimum_required (VERSION 2.6)
project (ARRAY)

# The version number.
set (ARRAY_VERSION_MAJOR 1)
set (ARRAY_VERSION_MINOR 0)


include(CheckCXXCompilerFlag)
check_cxx_compiler_flag (-std=c++0x HAVE_NEW_STD)

if (HAVE_NEW_STD)
  message (STATUS "Building shared library")
  add_definitions(-std=c++0x)
else()
  message(FATAL_ERROR "*** ERROR *** Compiler supporting the flag -std=c++0x is required to use the array library.")
endif()

find_package (Boost)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  set (EXTERNAL_LIBS ${Boost_LIBRARIES})
endif()


option (BUILD_SHARED_LIBS "Set to OFF to build static library" ON)
if (BUILD_SHARED_LIBS)
  message (STATUS "Building shared library")
  set (ARRAY_SHARED YES)
else()
  message (STATUS "Building static library")
  set (ARRAY_SHARED NO)
endif()
set (BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}" CACHE BOOL "Set to OFF to build static libraries" FORCE)


# verbose output
option (VERBOSE "Verbose output in ARRAY library" OFF)
message (STATUS "Setting ARRAY_VERBOSE variable to ${VERBOSE}")
if (VERBOSE)
  set (ARRAY_VERBOSE YES)
else ()
  set (ARRAY_VERBOSE NO)
endif ()


# add the binary tree to the search path for include files
include_directories(${PROJECT_BINARY_DIR})

# variable used to specify a user defined path for libraries
# and header files
set (ARRAY_USER_LIB_PATH "" CACHE PATH "User specified path for libraries")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

# cblas
message (STATUS "Looking for cblas header file")
find_path (CBLAS_INCLUDE NAMES cblas.h)
if (NOT CBLAS_INCLUDE)
  message (STATUS "*** WARNING *** Header file cblas.h not found.")
else()
  message (STATUS "*** INFO *** Path to cblas.h found:")
  message (STATUS "             ${CBLAS_INCLUDE}")
  set (HAVE_CBLAS_H YES)
  include_directories(${CBLAS_INCLUDE})
  message (STATUS "*** INFO *** Setting HAVE_CBLAS_H: ${HAVE_CBLAS_H}")
  message (STATUS "Looking for cblas library")
  find_library (CBLAS_LIB cblas)
  if (NOT CBLAS_LIB)
    message (STATUS "*** WARNING *** Library cblas not found.")
  else()
    message (STATUS "*** INFO *** Found cblas library:")
    message (STATUS "             ${CBLAS_LIB}")
    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CBLAS_LIB})
  endif()
endif()


# enable testing
enable_testing ()

add_subdirectory (array)
add_subdirectory (examples)
add_subdirectory (tests)

# Doxygen support
#-- Add an Option to toggle the generation of the API documentation
option(Documentation "Use Doxygen to create the HTML based API documentation" ON)

if(Documentation)

  FIND_PACKAGE(Doxygen)

  option (Doxygen_verbose "Verbose output in Doxygen" OFF)
  if (Doxygen_verbose)
    set (DOXYGEN_WARNING YES)
  else ()
    set (DOXYGEN_WARNING NO)
  endif ()

  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR 
      "*** ERROR *** Doxygen is needed to build the documentation. Please install it correctly")
  endif()

  message (STATUS "*** INFO *** Documentation will be built")

  option(LaTeX "Generate LaTeX documentation" ON)

  message (STATUS "*** INFO *** Documentation html directory: ${PROJECT_BINARY_DIR}/doc/html")

  if (LaTeX)
    set (DOXYFILE_GENERATE_LATEX YES)
    message (STATUS "*** INFO *** LaTeX documentation enabled")
    message (STATUS "*** INFO *** Documentation latex directory: ${PROJECT_BINARY_DIR}/doc/latex")
  else ()
    set (DOXYFILE_GENERATE_LATEX NO)
    message (STATUS "*** INFO *** LaTeX documentation disabled")  
  endif()
  
  set (DOXYFILE_SOURCE_DIRS ${CMAKE_SOURCE_DIR})
  set (CMAKE_IMAGE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/doc/figures)
  set (DOXYGEN_BIB_FILE ${PROJECT_SOURCE_DIR}/doc/bibliography.bib)
  set (PROJECT_VERSION "${ARRAY_VERSION_MAJOR}.${ARRAY_VERSION_MINOR}.${SUBVERSION_REVISION}")
  set (CMAKE_EXAMPLE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/examples)
  set (DOXYFILE_PDFLATEX YES)
  set (CMAKE_LATEX_PREAMBLE ${PROJECT_SOURCE_DIR}/cmake/LaTeX_preamble)

  #-- Configure the Template Doxyfile for our specific project
  configure_file(cmake/Doxyfile.in 
                 ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built 
  if (LaTeX)
    add_custom_target (doc 
            ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
                SOURCES ${PROJECT_BINARY_DIR}/Doxyfile
                COMMAND cd ${PROJECT_BINARY_DIR}/doc/latex && make && mv refman.pdf ../array.pdf)
  else ()
    add_custom_target (doc 
            ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
            SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
  endif()

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/doc)

  # IF you do NOT want the documentation to be generated EVERY time you build the project
  # then leave out the 'ALL' keyword from the above command.
endif()


set(CPACK_PACKAGE_VERSION_PATCH "${ARRAY_VERSION_MAJOR}_${ARRAY_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ;TZ")

INCLUDE(CPack)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  ${ARRAY_SOURCE_DIR}/cmake/array-config.hpp.in
  ${ARRAY_SOURCE_DIR}/ARRAY/array-config.hpp
  )
