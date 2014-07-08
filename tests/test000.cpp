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


/*! \file test000.cpp
 *
 * \brief This function tests indexed access through operator() and operator[].
 */


#include "array.hpp"

using std::cout;
using std::endl;

using array::vector_type;
using array::matrix_type;
using array::tensor_type;

int main() {
  
  size_t m = 16, n = 8, o = 4, p = 2;
  
  double *xdata = new double[m];
  for (size_t i=0; i<m; ++i)
    xdata[i] = i;
  vector_type<double> x(m, xdata);
  
  
  double *Adata = new double[m*n];
  for (size_t i=0; i<m*n; ++i)
    Adata[i] = i;
  matrix_type<double> A(m,n,Adata);
  
  
  double *TTdata = new double[m*n*o*p];
  for (size_t i=0; i<m*n*o*p; ++i)
    TTdata[i] = i;
  tensor_type<double> TT(m,n,o,p,TTdata);
  
  
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
