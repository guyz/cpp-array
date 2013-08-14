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
install (FILES  ${HEADER_FILES} DESTINATION include/array)

# add executable
add_executable(a.out main.cpp)

# link executable
target_link_libraries (a.out ${EXTERNAL_LIBS})
