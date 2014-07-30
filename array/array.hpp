/*
 * Copyright (©) 2014 Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
 * All Rights Reserved
 *
 * cpp-array is free  software: you can redistribute it and/or  modify it under
 * the terms  of the  GNU Lesser  General Public  License as  published by  the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * cpp-array is  distributed in the  hope that it  will be useful, but  WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A  PARTICULAR PURPOSE. See  the GNU  Lesser General  Public License  for
 * more details.
 *
 * You should  have received  a copy  of the GNU  Lesser General  Public License
 * along with cpp-array. If not, see <http://www.gnu.org/licenses/>.
 *
 */

/*! \file array.hpp
 *
 * \brief This file presents the implementaiton of the arbitrary-rank
 * tensor from a single Array class template.
 */

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "array_impl.hpp"
#include "functions.hpp"


#endif /* ARRAY_HPP */

////////////////////////////////////////////////////////////////////////////////
// documentation

/*!

 \verbatim
                                                                                               
    _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/  _/_/  _/  _/_/    _/_/_/  _/    _/   
 _/        _/    _/  _/    _/  _/_/_/_/_/  _/    _/  _/_/      _/_/      _/    _/  _/    _/    
_/        _/    _/  _/    _/              _/    _/  _/        _/        _/    _/  _/    _/     
 _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/        _/          _/_/_/    _/_/_/      
        _/        _/                                                                  _/       
       _/        _/                                                              _/_/          
 
 \endverbatim


 * \mainpage A C++ interface to the BLAS library using arbitrary-rank arrays
 * \author Alejandro Marcos Aragón, Ph.D.
 * \n Email: alejandro.aragon@fulbrightmail.org
 *
 * \date 06/05/2013
 * \note Preprint  <a href="http://arxiv.org/abs/1209.1003">arXiv:1209.1003</a>


 \section intro_sec Introduction


 The \c cpp-array project aims at creating a library of algebraic objects
 that interfaces to the \c BLAS set of functions to maintain high performance.
 The main aim of the library is to provide the end user with an easy to
 use syntax, where mathematical expressions can be used and the library
 takes care of calling the \c BLAS routines behind the scenes. This is done
 through the use of expression templates and features provided by the new
 standard library (C++11) so performance is not affected at all.
 Because \c cpp-array is an interface to the \c BLAS set of routines, a working
 \c BLAS implmenentation is needed.


 \section prereq Prerequisites

 The compilation of \c cpp-array requires the following packages:

 - \c CMake build system,
 - \c GNU \c make,
 - a C++ compiler that supports the C++11 set of requirements (tested
 on gcc 4.7 and clang 4.2, or newer versions of them),
 - An implementation of the \c BLAS set of routines (Fortran compiler needed) or
 its C interface (\c CBLAS),
 - \c Doxygen (for the documentation).

 \section config Configuration

 Unzip the \c cpp-array .zip package file. Change to the unzipped directory
using the command \c cd. The library can be configured in three different ways
from the command line. If a Fortran compiler is given by using the \c fc
parameter, the code tries to find a working Fortran BLAS implementation:
 \verbatim $ make config fc=gfortran \endverbatim

 If the user wants to use the Nvidia CUBLAS implementation instead to run the
code in GPUs, the library can be configured as follows:
 \verbatim $ make config cuda=1 \endverbatim

 If none of these parameters are given, the configure program will try to find a
C interface to the BLAS library:
 \verbatim $ make config \endverbatim

 The following table lists all the parameters that can be used to configure the
library:

 option               | default       | description
 ----------------     | ------------- | ------------
 \c cxx=[compiler]    | c++           | The C++ compiler to use
 \c fc=[compiler]     | None          | The Fortran compiler to use
 \c cuda=[bool]       | false         | Enable GPU computing
 \c prefix=[path]     | /usr/local    | Installation path
 \c doc=[bool]        | true          | Build the documentation
 \c latex=[bool]      | false         | Enable LaTeX documentation
 \c build=[string]    | None          | Build type (Debug, Release, etc.)


 \note There is no default as the library can be built without Fortran support.


 The variables resulting from the configuration can be edited direcly by typing
 \verbatim $ make edit_cache \endverbatim

 This may prove useful, \e e.g., in the case \c CMake fails to find the \c CBLAS
 library because it is located on a non-standard directory.

 \section compil Compilation


 To compile the library once the configuration has taken place, type
 \verbatim $ make \endverbatim

 All programs are stored within the directory build/\<architecture\>/, where
 \<architecture\> is machine dependent (\e e.g., Linux-x86_64, or
Darwin-x86_64).

 \section install Installation


 Install the library by typing
 \verbatim $ make install \endverbatim

 \section other Other

 To remove the installed files, type
 \verbatim $ make uninstall \endverbatim

 To remove all object files while retaining the configuration, type
 \verbatim $ make clean \endverbatim

 To remove the build directory, type
 \verbatim $ make distclean \endverbatim

 The documentation for the library can be built by running
 \verbatim $ make doc \endverbatim
 assuming that \c Doxygen is installed in the system. The documentation is
 also generated within the directory build/\<architecture\>/.

 The \c examples/ folder shows how to use the library with a set of examples.
 Check the source files in this folder for a correct usage of the library.
 To build the examples, type
 \verbatim $ make examples \endverbatim

 To run the examples, change to the directory build/\<architecture\>/examples
 and execute the corresponding program.

 The tests in the \c tests/ directory are run to make sure that the library is
 built correctly. If any of the tests fails, check the difference between the
 .\c lastout and the .\c verified files for that particular test in order to
find
 the problem. The tests are executed by the \c CTest framework, by typing
 \verbatim $ make check \endverbatim

 \section devel For developers


 The \c CMake build system can be accessed directly to have better control on
the
 build options. By running the command
 \verbatim $ make edit_cache \endverbatim
 the user can customize many more options than those presented above.

 */
