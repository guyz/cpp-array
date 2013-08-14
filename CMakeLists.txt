# The name of our project is cpp-array. CMakeLists files in this project can 
# refer to the root source directory of the project as ${cpp-array_SOURCE_DIR}
# and to the root binary directory of the project as ${cpp-array_BINARY_DIR}

cmake_minimum_required (VERSION 2.6)
project (cpp-array)

# The version number.
set (ARRAY_VERSION_MAJOR 1)
set (ARRAY_VERSION_MINOR 0)

set (cpp-array_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/array)


# set path for additional cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


include(CheckCXXCompilerFlag)
check_cxx_compiler_flag (-std=c++0x HAVE_NEW_STD)

if (HAVE_NEW_STD)
  message (STATUS "Compiler supports C++11 set of requirements.")
  add_definitions(-std=c++0x)
else()
  message(WARNING "*** ERROR *** Compiler supporting the flag -std=c++0x is required to use the array library.")
endif()

# add definitions if clang compiler
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  message (STATUS "*** INFO *** Adding -stdlib=libc++ to Clang compiler")
  add_definitions(-stdlib=libc++)
  set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS_MINSIZE "-stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS_RELWITH "-stdlib=libc++")
  set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++") # added to search for unordered set and map
endif()


# add the binary tree to the search path for include files
list (APPEND cpp-array_INCLUDE_DIRS ${PROJECT_BINARY_DIR})

# variable used to specify a user defined path for libraries
# and header files
set (ARRAY_USER_LIB_PATH "" CACHE PATH "User specified path for libraries")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)



option (CUDA "Use cublas implementation" OFF)

if (CUDA)

  message(STATUS "*** INFO *** CUDA option enabled")
  
  find_package(CUDA)
  if (CUDA_FOUND)

    message(STATUS "*** INFO *** Path to cublas.h")
    message(STATUS "             ${CUDA_INCLUDE_DIRS}")
    set (HAVE_CUBLAS_H YES)
    list (APPEND cpp-array_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

    message(STATUS "*** INFO *** Found cublas library:")
    message(STATUS "             ${CUDA_CUBLAS_LIBRARIES}")
    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})

  endif()
endif()


if (NOT CUDA_FOUND)

  # find CBLAS library
  find_package(CBLAS)
  if(CBLAS_FOUND)
    set (HAVE_CBLAS_H YES)
    message (STATUS "  Setting HAVE_CBLAS_H: ${HAVE_CBLAS_H}")
    list (APPEND cpp-array_INCLUDE_DIRS ${CBLAS_INCLUDE_DIRS})
    message (STATUS "  Adding include directories: ${CBLAS_INCLUDE_DIRS}")
    set (EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CBLAS_LIBRARIES})
    message (STATUS "  Adding CBLAS libraries: ${CBLAS_LIBRARIES}")
  else()
    message (STATUS "*** WARNING *** CBLAS library not found.")
  endif()
endif()


include_directories(${cpp-array_INCLUDE_DIRS})

set (cpp-array_INCLUDE_DIRS_TMP ${cpp-array_INCLUDE_DIRS})
set (cpp-array_INCLUDE_DIRS ${cpp-array_INCLUDE_DIRS_TMP} CACHE INTERNAL "Include directories for cpp-array" FORCE)

message (STATUS ${cpp-array_INCLUDE_DIRS})


# enable testing
enable_testing ()

add_subdirectory (array)
add_subdirectory (examples)
add_subdirectory (tests)


# Doxygen support
#-- Add an Option to toggle the generation of the API documentation
option(cpp-array_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)

if(cpp-array_DOCUMENTATION)

  find_package(Doxygen)

  option (Doxygen_verbose "Verbose output in Doxygen" OFF)
  if (Doxygen_verbose)
    set (DOXYGEN_WARNING YES)
  else ()
    set (DOXYGEN_WARNING NO)
  endif ()

  if (NOT DOXYGEN_FOUND)
    message(WARNING 
      "*** WARNING *** Doxygen is needed to build the documentation. Please install it correctly")
  else()

    message (STATUS "*** INFO *** Documentation enabled.")

    option(LaTeX "Generate LaTeX documentation" OFF)

    message (STATUS "*** INFO *** Documentation html directory: ${PROJECT_BINARY_DIR}/doc/html")

    if (LaTeX)
      set (DOXYFILE_GENERATE_LATEX YES)
      message (STATUS "*** INFO *** LaTeX documentation enabled")
      message (STATUS "*** INFO *** Documentation latex directory: ${PROJECT_BINARY_DIR}/doc/latex")
    else ()
      set (DOXYFILE_GENERATE_LATEX NO)
      message (STATUS "*** INFO *** LaTeX documentation disabled")  
    endif()
  
    set (DOXYFILE_SOURCE_DIRS ${PROJECT_SOURCE_DIR})
    set (CMAKE_IMAGE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/doc/figures)
    set (DOXYGEN_BIB_FILE ${PROJECT_SOURCE_DIR}/doc/bibliography.bib)
    set (PROJECT_VERSION "${YAFEQ_VERSION_MAJOR}.${YAFEQ_VERSION_MINOR}.${SUBVERSION_REVISION}")
    set (CMAKE_EXAMPLE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/examples)
    set (DOXYFILE_PDFLATEX YES)
    set (CMAKE_LATEX_PREAMBLE ${PROJECT_SOURCE_DIR}/cmake/LaTeX_preamble)
    set (PROJECT_BRIEF "Yet Another Finite Element Qode")
  
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

  message (STATUS "*** INFO *** Documentation disabled.")

endif()




SET(VERSION "${ARRAY_VERSION_MAJOR}.${ARRAY_VERSION_MINOR}")

# cpack package module
SET(CPACK_INSTALL_CMAKE_PROJECTS "${cpp-array_BINARY_DIR};cpp-array;ALL;/")
SET(CPACK_NSIS_DISPLAY_NAME "cpp-array ${VERSION}")
SET(CPACK_OUTPUT_CONFIG_FILE "${cpp-array_BINARY_DIR}/CPackConfig.cmake")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${cpp-array_SOURCE_DIR}/README")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A C++ interface to the BLAS library using arbitrary-rank arrays")
SET(CPACK_PACKAGE_FILE_NAME "cpp-array-${VERSION}-${CMAKE_SYSTEM_NAME}")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "cpp-array ${VERSION}")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "cpp-array ${VERSION}")
SET(CPACK_PACKAGE_NAME "cpp-array")
SET(CPACK_PACKAGE_VERSION "${VERSION}")
SET(CPACK_PACKAGE_VERSION_MAJOR "${ARRAY_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${ARRAY_VERSION_MINOR}")
SET(CPACK_SOURCE_IGNORE_FILES "/build/;/.svn/;.DS_Store;.#;/#;.*~;.*xcodeproj")
SET(CPACK_RESOURCE_FILE_README "${cpp-array_SOURCE_DIR}/README")
SET(CPACK_RESOURCE_FILE_WELCOME "${cpp-array_SOURCE_DIR}/README")
SET(CPACK_GENERATOR "ZIP")
SET(CPACK_SOURCE_GENERATOR "ZIP")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "cpp-array-${VERSION}-src")
SET(CPACK_SOURCE_STRIP_FILES "")
SET(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
SET(CPACK_TOPLEVEL_TAG "${CMAKE_SYSTEM_NAME}")


INCLUDE(CPack)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  ${cpp-array_SOURCE_DIR}/cmake/array-config.hpp.in
  ${cpp-array_SOURCE_DIR}/array/array-config.hpp
  )
