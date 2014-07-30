%module cpparray

%{
#include "array_impl.hpp"
  %}
// Try grabbing it unmodified
%include "std_iostream.i"
%include "array_impl.hpp"

//namespace array {

//  %template(MatrixDouble) Array<2,double>;
//  %include "array.hpp"

//};


%init %{
  
%}

//#include <stdexcept>
//#include <iostream>
//#include <type_traits>
//#include "array-config.hpp"
//using namespace array;
//#define SWIG_FILE_WITH_INIT
//%include "std_except.i"


// Try grabbing it unmodified
//%include <type_traits>
//%include "array-config.hpp"
