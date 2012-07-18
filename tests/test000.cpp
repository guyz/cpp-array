/*
 * Copyright (C) 2012 by Alejandro M. Aragón
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

#include "array.hpp"

using std::cout;
using std::endl;

using array::vector_type;
using array::matrix_type;
using array::tensor_type;

int main() {
    
    size_t m = 16, n = 8, o = 4, p = 2;
    
    vector_type<double> x(m);
    matrix_type<double> A(m,n);
    tensor_type<double> TT(m,n,o,p);
    
    for (size_t i=0; i<m; ++i)
        x.data_[i] = i;
    
    for (size_t i=0; i<m*n; ++i)
        A.data_[i] = i;
    
    for (size_t i=0; i<m*n*o*p; ++i)
        TT.data_[i] = i;
    
    cout<<"Testing array access...";
    
    for (size_t i=0; i<m; ++i) {
        
        // test vector
        assert(x(i) == x[i]);
        assert(x(i) == i);
        
        for (size_t j=0; j<n; ++j) {
            
            // test matrix
            assert(A(i,j) == A[i][j]);
            assert(A(i,j) == i + m*j);
            
            for (size_t k=0; k<o; ++k)
                for (size_t l=0; l<p; ++l) {
                    
                    // test tensor
                    assert(TT(i,j,k,l) == TT[i][j][k][l]);
                    assert(TT(i,j,k,l) == i + m*j + m*n*k + m*n*o*l);
                }
        }
    }
    
    cout<<"Ok!"<<endl;
    
    return 0;
}  
