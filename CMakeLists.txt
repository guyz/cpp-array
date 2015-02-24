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


# The name of our project is cpp-array. CMakeLists files in this project can
# refer to the root source directory of the project as ${CPP-ARRAY_SOURCE_DIR}
# and to the root binary directory of the project as ${CPP-ARRAY_BINARY_DIR}

cmake_minimum_required (VERSION 2.8)
project (CPP-ARRAY)
enable_language(CXX)


# set path for additional cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


# The version number.
set (ARRAY_VERSION_MAJOR 1)
set (ARRAY_VERSION_MINOR 1)

# check if there is a working copy of the repository
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.svn")

  # Subversion
  find_package(Subversion)
  if(Subversion_FOUND)
    Subversion_WC_INFO(${CMAKE_CURRENT_SOURCE_DIR} ER)
    set(VERSION "${ARRAY_VERSION_MAJOR}.${ARRAY_VERSION_MINOR}.${ER_WC_REVISION}")
  endif(Subversion_FOUND)
  
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")

  include(GetGitRevisionDescription)
  git_describe(VERSION --tags)

  #parse the version information into pieces.
  string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
  string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_SHA1 "${VERSION}")
  set(VERSION_SHORT "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

else()
  set(VERSION "${ARRAY_VERSION_MAJOR}.${ARRAY_VERSION_MINOR}")
endif()

message (STATUS "Configuring cpp-array ${VERSION}")


# check if Fortran compiler in the environment
if (DEFINED ENV{fc} OR DEFINED ENV{FC})
  message(STATUS "Fortran compiler given: $ENV{fc}$ENV{FC}")
  enable_language(Fortran)
else()
  message(STATUS "Fortran compiler not provided:")
  if (DEFINED ENV{cuda})
    message(STATUS "  Configuring the library with CUDA support")
  else()
    message(STATUS "  Configuring the library with the C interfaces to BLAS and LAPACK")
  endif()
endif()

# check for compiler supporting C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag (-std=c++0x HAVE_NEW_STD)

if (HAVE_NEW_STD)
  message (STATUS "Compiler supports C++11 set of requirements.")
  add_definitions(-std=c++0x)
#  add_definitions(-std=c++11)
else()
  message(WARNING "*** ERROR *** Compiler supporting the flag -std=c++0x is required to use the array library.")
endif()



set (CPP-ARRAY_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/array)


# add the binary tree to the search path for include files
list (APPEND CPP-ARRAY_INCLUDE_DIRS ${PROJECT_BINARY_DIR})

# variable used to specify a user defined path for libraries
# and header files
set (ARRAY_USER_LIB_PATH "" CACHE PATH "User specified path for libraries")

#include(CheckIncludeFile)
include(CheckIncludeFileCXX)


# enable GPU computing through the CUDA library
option (CUDA "Use cublas implementation" OFF)

if (CUDA)

  message(STATUS "CUDA option enabled")
  
  find_package(CUDA)
  if (CUDA_FOUND)

    message(STATUS "Path to cublas.h: ${CUDA_INCLUDE_DIRS}")
    set (HAVE_CUBLAS_H YES)
    list (APPEND CPP-ARRAY_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    message(STATUS "Found cublas library: ${CUDA_CUBLAS_LIBRARIES}")
    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})

  endif()
endif()

# use blas or cblas if not working with cuda
if (NOT CUDA_FOUND)

  find_package(BLAS)
  if (BLAS_FOUND AND CMAKE_Fortran_COMPILER)

    set (HAVE_BLAS_H YES)
    message (STATUS "  Setting HAVE_BLAS_H: ${HAVE_BLAS_H}")
    enable_language(Fortran)

    include(FortranCInterface)
    FortranCInterface_HEADER(${CMAKE_CURRENT_BINARY_DIR}/fortran_mangling.hh MACRO_NAMESPACE "CPPARRAY_FC_")

    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${BLAS_LIBRARIES})
    message (STATUS "  Adding BLAS libraries: ${BLAS_LIBRARIES}")

  else()

    # find CBLAS library
    find_package(CBLAS)
    if(CBLAS_FOUND)
      set (HAVE_CBLAS_H YES)
      message (STATUS "  Setting HAVE_CBLAS_H: ${HAVE_CBLAS_H}")
      message (STATUS "  Setting CBLAS_HEADER: ${CBLAS_HEADER}")
      list (APPEND CPP-ARRAY_INCLUDE_DIRS ${CBLAS_INCLUDE_DIRS})
      message (STATUS "  Adding include directories: ${CBLAS_INCLUDE_DIRS}")
      set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CBLAS_LIBRARIES})
      message (STATUS "  Adding CBLAS libraries: ${CBLAS_LIBRARIES}")
    else()
      message (STATUS "*** WARNING *** CBLAS library not found.")
    endif()
  endif()
endif()


# find lapack library
if (CMAKE_Fortran_COMPILER)

  find_package(LAPACK)
  if(LAPACK_FOUND OR LAPACK95_FOUND)
    set (HAVE_LAPACK YES)
    message (STATUS "  Setting HAVE_LAPACK: ${HAVE_LAPACK}")

    enable_language(Fortran)
    include(FortranCInterface)
    FortranCInterface_HEADER(${CMAKE_CURRENT_BINARY_DIR}/fortran_mangling.hh MACRO_NAMESPACE "CPPARRAY_FC_")

    if(LAPACK_FOUND)
      set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${LAPACK_LIBRARIES})
      message (STATUS "  Adding LAPACK libraries: ${LAPACK_LIBRARIES}")
    elseif (LAPACK95_FOUND)
      set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${LAPACK95_LIBRARIES})
      message (STATUS "  Adding LAPACK libraries: ${LAPACK95_LIBRARIES}")
    endif()
  else()
    message (STATUS "  LAPACK library not found")
  endif()

else(CMAKE_Fortran_COMPILER)

  # find CLAPACK library
  find_package(CLAPACK)
  if(CLAPACK_FOUND)
    set (HAVE_CLAPACK YES)
    message (STATUS "  Setting HAVE_CLAPACK: ${HAVE_CLAPACK}")

    message (STATUS "  Setting CLAPACK_HEADER: ${CLAPACK_HEADER}")
    list (APPEND CPP-ARRAY_INCLUDE_DIRS ${CLAPACK_INCLUDE_DIRS})
    message (STATUS "  Adding include directories: ${CLAPACK_INCLUDE_DIRS}")
    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CLAPACK_LIBRARIES})
    message (STATUS "  Adding CLAPACK libraries: ${CLAPACK_LIBRARIES}")
  else()
    message (STATUS "*** WARNING *** LAPACK library not found.")
  endif()
endif (CMAKE_Fortran_COMPILER)


include_directories(${CPP-ARRAY_INCLUDE_DIRS})

set (CPP-ARRAY_INCLUDE_DIRS_TMP ${CPP-ARRAY_INCLUDE_DIRS})
set (CPP-ARRAY_INCLUDE_DIRS ${CPP-ARRAY_INCLUDE_DIRS_TMP} CACHE INTERNAL "Include directories for cpp-array" FORCE)

# set external libraries
set (CPP-ARRAY_LIBRARIES ${EXTERNAL_LIBS} CACHE INTERNAL "Libraries for cpp-array" FORCE)

message (STATUS ${CPP-ARRAY_INCLUDE_DIRS})

add_subdirectory (array)

option (CPP-ARRAY_EXAMPLES "cpparray examples" ON)

if (CPP-ARRAY_EXAMPLES)
  # enable examples
  add_subdirectory (examples)
endif()

option (CPP-ARRAY_TESTS "cpparray tests" ON)

if (CPP-ARRAY_TESTS)
  # enable testing
  enable_testing ()
  add_subdirectory (tests)
endif()


# Doxygen support
#-- Add an Option to toggle the generation of the API documentation
option(CPP-ARRAY_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" ON)

if(CPP-ARRAY_DOCUMENTATION)

  find_package(Doxygen)

  if (NOT DOXYGEN_FOUND)
    message(WARNING 
      "*** WARNING *** Doxygen is needed to build the documentation. Please install it correctly")
  else()


    set (DOXYGEN_WARNING YES)
    message (STATUS "Documentation enabled.")
    message (STATUS "Found Doxygen: ${DOXYGEN_EXECUTABLE}")

    option (LaTeX "Generate LaTeX documentation" OFF)

    message (STATUS "Documentation html directory: ${PROJECT_BINARY_DIR}/doc/html")
    if (LaTeX)
      set (DOXYFILE_GENERATE_LATEX YES)
      message (STATUS "LaTeX documentation enabled")
      message (STATUS "Documentation latex directory: ${PROJECT_BINARY_DIR}/doc/latex")
    else ()
      set (DOXYFILE_GENERATE_LATEX NO)
      message (STATUS "LaTeX documentation disabled")  
    endif()
  
    set (DOXYFILE_SOURCE_DIRS ${PROJECT_SOURCE_DIR})
    set (PROJECT_VERSION "${VERSION}")
    set (CMAKE_EXAMPLE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/examples)
    set (DOXYFILE_PDFLATEX YES)
    set (CMAKE_LATEX_PREAMBLE ${PROJECT_SOURCE_DIR}/cmake/LaTeX_preamble)
    set (PROJECT_BRIEF "A C++ interface to the BLAS library using arbitrary-rank arrays")
  
    #-- Configure the Template Doxyfile for our specific project
    configure_file(cmake/Doxyfile.in 
                   ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
    #-- Add a custom target to run Doxygen when ever the project is built 
    if (LaTeX)
    
      add_custom_target (doc 
              ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
                  SOURCES ${PROJECT_BINARY_DIR}/Doxyfile
                  COMMAND make -C ${PROJECT_BINARY_DIR}/doc/latex && mv ${PROJECT_BINARY_DIR}/doc/latex/refman.pdf ${PROJECT_BINARY_DIR}/doc/cpp-array.pdf)
    else ()
      add_custom_target (doc 
              ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
              SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
    endif()

    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/doc)

    # IF you do NOT want the documentation to be generated EVERY time you build the project
    # then leave out the 'ALL' keyword from the above command.
  endif()
  
else()

  message (STATUS "Documentation disabled.")

endif()


# cpack package module
set(CPACK_INSTALL_CMAKE_PROJECTS "${PROJECT_BINARY_DIR};cpp-array;ALL;/")
set(CPACK_NSIS_DISPLAY_NAME "cpp-array ${VERSION_SHORT}")
set(CPACK_OUTPUT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackConfig.cmake")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A C++ interface to the BLAS library using arbitrary-rank arrays")
set(CPACK_PACKAGE_FILE_NAME "cpp-array-${VERSION_SHORT}-${CMAKE_SYSTEM_NAME}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "cpp-array ${VERSION_SHORT}")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "cpp-array ${VERSION_SHORT}")
set(CPACK_PACKAGE_NAME "cpp-array")
set(CPACK_PACKAGE_VERSION "${VERSION_SHORT}")
set(CPACK_PACKAGE_VERSION_MAJOR "${ARRAY_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ARRAY_VERSION_MINOR}")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.svn/;.git/;.gitignore;.DS_Store;.#;/#;.*~;.*xcodeproj;project.*;xcuserdata")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/README")
set(CPACK_GENERATOR "ZIP")
set(CPACK_SOURCE_GENERATOR "ZIP;TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "cpp-array-${VERSION_SHORT}-src")
set(CPACK_SOURCE_STRIP_FILES "")
set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
set(CPACK_TOPLEVEL_TAG "${CMAKE_SYSTEM_NAME}")
include(CPack)




# swig module creation for python

# hard-code 3.4 for the time being
#set(PYTHON_INCLUDE_PATH /Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m)
#set(PYTHON_LIBRARIES /Library/Frameworks/Python.framework/Versions/3.4/lib/libpython3.4.dylib)
#set(PYTHON_EXECUTABLE /usr/local/bin/python3)
#set(PYTHON_INCLUDE_DIR  /Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m)


#find_package(SWIG)
if(SWIG_FOUND)
  
  include(${SWIG_USE_FILE})

#  find_package(PythonLibs 3)
  include_directories(${PYTHON_INCLUDE_PATH})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/array)
  
  message(STATUS "PYTHON LIBS: ${PYTHON_LIBRARIES}")

  ##set(CMAKE_SWIG_FLAGS "-c++ -shadow")
  set(CMAKE_SWIG_FLAGS "-c++;-shadow")


  SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/array/array.i PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(array.i PROPERTIES SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(cpparray python ${CMAKE_CURRENT_SOURCE_DIR}/array/array.i)
  SWIG_LINK_LIBRARIES(cpparray ${PYTHON_LIBRARIES} ${EXTERNAL_LIBS})
endif()


# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  ${PROJECT_SOURCE_DIR}/cmake/array-config.hpp.in
  ${PROJECT_BINARY_DIR}/array-config.hpp
)

# configure file to use when find_package is used with cpp-array
configure_file (
  ${PROJECT_SOURCE_DIR}/cmake/cpp-array-config.cmake.in
  ${PROJECT_BINARY_DIR}/cpp-array-config.cmake
  @ONLY
)

install (FILES ${PROJECT_BINARY_DIR}/cpp-array-config.cmake DESTINATION include/array)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/fortran_mangling.hh)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fortran_mangling.hh DESTINATION include)
endif()