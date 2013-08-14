/*
 * Copyright (C) 2013 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

//! \file performance.cpp
//
// This file can be used to test the performance of the array in the
// context of matrix-matrix multiplication. The comparison is carried
// out directly by calling a BLAS routine.
//

#include "array-config.hpp"

#include "expr.hpp"


#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>


using std::cout;
using std::endl;
using std::setw;
using namespace std::chrono;


int main() {
  
  
  cout<<setw(20)<<"size"<<setw(20)<<"blas"<<setw(20)<<"array"<<endl;
  
  // loop over matrix sizes
  for (size_t m=125; m<= 16000; m *= 2) {
    
    
    array::matrix_type<double> A(m,m);
    array::matrix_type<double> B(m,m);
    
    int k = 0;
    int l = 0;
    for (size_t i=0; i<m; ++i)
      for (size_t j=0; j<m; ++j) {
        A(i,j) = ++k;
        B(i,j) = --l;
      }
    
    // compute time using array
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    array::matrix_type<double> C = A*B;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
      
    // compute time using blas
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    array::matrix_type<double> D(A.rows(), B.columns());
    array::cblas_gemm(CblasNoTrans, CblasNoTrans, D.rows(), D.columns(),
                      A.columns(), 1., A.data(), A.rows(), B.data(), B.rows(), 1.0, D.data(), D.rows());
    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    // compute elapsed time
    duration<double> time_array = duration_cast<duration<double>>(t2 - t1);
    duration<double> time_blas = duration_cast<duration<double>>(t4 - t3);

    // print results
    cout<<setw(20)<<m<<setw(20)<<time_array.count()<<setw(20)<<time_blas.count()<<endl;
    
  } // loop over matrix sizes

  
  return 0;
}
