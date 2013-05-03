/*
 * Copyright (C) 2011 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@gmail.com>
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
//  Created by Alejandro Aragón on 10/10/11.
//

#include "array-config.hpp"

#define ARRAY_VERBOSE_TMP ARRAY_VERBOSE
#ifdef ARRAY_VERBOSE
#undef ARRAY_VERBOSE
#endif

#include "expr.hpp"
#include "timer.hpp"


using std::cout;
using std::endl;


int main() {
    
    size_t m = 16000;
    size_t iter = 1;
    
    array::matrix_type<double> A(m,m);
    array::matrix_type<double> B(m,m);
    
    int k = 0;
    int l = 0;
    for (size_t i=0; i<m; ++i)
        for (size_t j=0; j<m; ++j) {
            A(i,j) = ++k;
            B(i,j) = --l;
        }
    
    array::ctimer t;
    
    
    for (size_t i=0; i<iter; ++i) {
      
    array::matrix_type<double> C = A*B;
      
//        array::matrix_type<double> C(A.rows(), B.columns());
//        array::cblas_gemm(CblasNoTrans, CblasNoTrans, C.rows(), C.columns(),
//                          A.columns(), 1., A.data_, A.rows(), B.data_, B.rows(), 1.0, C.data_, C.rows());
    }
    
    cout<<t<<endl;
    
  
    return 0;
}
