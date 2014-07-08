/*
 * Copyright (©) 2014 Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
 * All Rights Reserved
 *
 * cpp-array is free  software: you can redistribute it and/or  modify it under
 * the terms  of the  GNU Lesser  General Public  License as  published by  the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
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

//! \example dot_performance.cpp
//
// This file can be used to test the performance of the array in the
// context of scalar product. The comparison is carried out directly by
// calling a BLAS routine.
//

#include "array-config.hpp"

#include "expr.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>


using std::cout;
using std::endl;
using std::setw;
using namespace std::chrono;


int main() {
  
  typedef steady_clock clock_type;
  
  size_t iter = 10;
  size_t w1 = 12;
  size_t w2 = 24;
  
  cout<<setw(w1)<<"size"<<setw(w2)<<"array"<<setw(w2)<<"blas"<<endl;
  
  
  // loop over matrix sizes
  for (size_t m=1000; m<=524288000; m *= 2) {
    
    array::vector_type<double> x(m,1.);
    array::vector_type<double> y(m,1.);
    
    // compute time using array
    clock_type::time_point t1 = clock_type::now();
    for (size_t i=0; i<iter; ++i)
      double r = transpose(x)*y;
    clock_type::time_point t2 = clock_type::now();
    
    
    // compute time using blas
    clock_type::time_point t3 = clock_type::now();
    for (size_t i=0; i<iter; ++i)
      double r = array::cblas_dot<double>(x.size(), x.data(), 1, y.data(), 1);
    
    clock_type::time_point t4 = clock_type::now();
    
    // compute elapsed time
    auto ms1 = duration_cast<microseconds>(t2 - t1)/iter;
    auto ms2 = duration_cast<microseconds>(t4 - t3)/iter;
    
    // print results
    minutes mm1 = duration_cast<minutes>(ms1 % hours(1));
    seconds ss1 = duration_cast<seconds>(ms1 % minutes(1));
    milliseconds msec1 = duration_cast<milliseconds>(ms1 % seconds(1));
    
    minutes mm2 = duration_cast<minutes>(ms2 % hours(1));
    seconds ss2 = duration_cast<seconds>(ms2 % minutes(1));
    milliseconds msec2 = duration_cast<milliseconds>(ms2 % seconds(1));
    
//    std::stringstream sstr;
//    sstr<<mm1.count()<<"m "<<ss1.count()<<"s "<<msec1.count()<<"ms "<<(ms1 % milliseconds(1)).count()<<"microseconds";
//    cout<<setw(w1)<<m<<setw(w2)<<sstr.str();
//    
//    sstr.str("");
//    sstr<<mm2.count()<<"m "<<ss2.count()<<"s "<<msec2.count()<<"ms "<<(ms2 % milliseconds(1)).count()<<"microseconds";
//    cout<<setw(w2)<<sstr.str()<<endl;
    
    std::stringstream sstr;
    sstr<<ms1.count();
    cout<<setw(w1)<<m<<setw(w2)<<sstr.str();
    
    sstr.str("");
    sstr<<ms2.count();
    cout<<setw(w2)<<sstr.str()<<endl;

    
  } // loop over matrix sizes
  
  return 0;
}
