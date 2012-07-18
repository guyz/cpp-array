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

#include "expr.hpp"
#include "typelist.hpp"
#include "timer.hpp"

#include <iostream>

using std::cout;
using std::endl;

using array::vector_type;
using array::matrix_type;
using array::tensor_type;


int main() {
  
  
  //    array::ExprIdentity<double> aa;
  //    array::ExprIdentity<double> bb;
  array::ExprIdentity<double> xx;
  array::ExprLiteral<double> yy(2.);
  array::ExprLiteral<int> zz(2);
  //    evaluate((2+x)/(1.*y), 0.0, 10.0);
  
  cout<<"2+yy = "<<(2+yy)<<endl;
  cout<<"zz+yy = "<<(zz+yy)<<endl;
  cout<<"2+xx(2) = "<<(2+xx(2))<<endl;
  cout<<"yy+2 = "<<(yy+2)<<endl;
  cout<<"xx(2)+2 = "<<(xx(2)+2)<<endl;
  
  cout<<"2-yy = "<<(2-yy)<<endl;
  cout<<"2-xx(2) = "<<(2-xx(2))<<endl;
  cout<<"yy-2 = "<<(yy-2)<<endl;
  cout<<"xx(2)-2 = "<<(xx(2)-2)<<endl;
  
  cout<<"2*yy = "<<(2*yy)<<endl;
  cout<<"2*xx(2) = "<<(2*xx(2))<<endl;
  cout<<"yy*2 = "<<(yy*2)<<endl;
  cout<<"xx(2)*2 = "<<(xx(2)*2)<<endl;
  
  cout<<"2/yy = "<<(2/yy)<<endl;
  cout<<"2/xx(2) = "<<(2/xx(2))<<endl;
  cout<<"yy/2 = "<<(yy/2)<<endl;
  cout<<"xx(2)/2 = "<<(xx(2)/2)<<endl;
  
  size_t m = 3, n = 3;
  
  vector_type<double> X(m), Y(m);
  
  for (int i=0; i<m; ++i) {
    X(i) = i;
    Y(i) = m-i;
  }
    
  cout<<"X -> "<<X<<endl;
  cout<<"Y -> "<<Y<<endl;
  
  cout<<"X + Y = "<<(X+Y)<<endl;
  cout<<"X - Y = "<<(X-Y)<<endl;
  
  cout<<"2.*X = "<<(2.*X)<<endl;
  cout<<"(2.*X)+Y = "<<((2.*X)+Y)<<endl;
  cout<<"X+(2*Y) = "<<(X+(2*Y))<<endl;
  cout<<"(X+(2*Y))+Y = "<<((X+(2*Y))+Y)<<endl;
  cout<<"Y+(X+(2*Y)) = "<<(Y+(X+(2*Y)))<<endl;
  
  cout<<"(2.*X)+(3*Y) = "<<((2.*X)+(3*Y))<<endl;
  
  cout<<(transpose(X) * Y)<<endl;
  cout<<(X * transpose(Y))<<endl;
  cout<<((2.*X) * transpose(Y))<<endl;
  cout<<(X * (2*transpose(Y)))<<endl;
  
  cout<<(X*(2.*(2.*transpose(Y))))<<endl;
  cout<<(X*((transpose(Y)*2.)*2.))<<endl;
  
  matrix_type<double> A(m,m);
  matrix_type<double> B(m,m);
  
  int k=0;
  int l=-1;
  for (int i=0; i<m; ++i)
    for (int j=0; j<m; ++j) {
      A(i,j) = ++k;
      B(i,j) = --l;
    }
  
  
  (A - (X * transpose(Y)));
  
  cout<<" A "<<A<<endl;
  matrix_type<double> Atr = transpose(A);
  cout<<" Atr = A' = "<<Atr<<endl;
  
  // matrix addition
  cout<<"A+A: "<<(A+A)<<endl;
  cout<<"A+(2.*A): "<<(A+(2.*A))<<endl;
  cout<<"(2.*A)+A: "<<((2.*A)+A)<<endl;
  cout<<"(2.*A)+(2.*A): "<<((2.*A)+(2.*A))<<endl;
  
  // matrix subtraction
  cout<<"A-A: "<<(A-A)<<endl;
  cout<<"A-(2.*A): "<<(A-(2.*A))<<endl;
  cout<<"(2.*A)-A: "<<((2.*A)-A)<<endl;
  cout<<"(2.*A)-(2.*A): "<<((2.*A)-(2.*A))<<endl;
  
  // matrix multiplication
  cout<<"A*A: "<<(A*A)<<endl;
  cout<<"A*(2.*A): "<<(A*(2.*A))<<endl;
  cout<<"(2.*A)*A: "<<((2.*A)*A)<<endl;
  cout<<"(2.*A)*(2.*A): "<<((2.*A)*(2.*A))<<endl;
  
  // matrix - vector multiplication
  cout<<"A*X: "<<(A*X)<<endl;
  cout<<"A*(2.*X): "<<(A*(2.*X))<<endl;
  cout<<"(2.*A)*X): "<<((2.*A)*X)<<endl;
  cout<<"(2.*A)*(3.*X): "<<(2.*A)*(3.*X)<<endl;
  
  // support for reference operators
  matrix_type<double> C = A;
  
  cout<<"A: "<<A<<endl;
  cout<<"B: "<<B<<endl;
  cout<<"C: "<<C<<endl;
  C += A;
  cout<<"C += A = "<<C<<endl;
  C += 2.*A;
  cout<<"C += 2.*A = "<<C<<endl;
  C += 2.*A*B;
  cout<<"C += 2.*A*B = "<<C<<endl;
  
  C -= 2.*A*B;
  cout<<"C -= 2.*A*B = "<<C<<endl;
  C -= 2.*A;
  cout<<"C -= 2*A = "<<C<<endl;
  C -= A;
  cout<<"C -= A = "<<C<<endl;
  
  // support for transposed objects
  
  m = 5;
  n = 3;
  A = matrix_type<double>(m,n);
  B = matrix_type<double>(n,m);
  k=0;
  l=-1;
  for (int i=0; i<m; ++i)
    for (int j=0; j<n; ++j) {
      A(i,j) = ++k;
      B(j,i) = --l;
    }
  
  cout<<"A: "<<A<<endl;
  cout<<"B: "<<B<<endl;
  cout<<"C: "<<C<<endl;
  
  cout<<"transpose(B)*C: "<<(transpose(B)*C)<<endl;
  cout<<"(2.*transpose(B))*C: "<<(2.*transpose(B)*C)<<endl;
  cout<<"transpose(B)*(2.*C): "<<(transpose(B)*(2.*C))<<endl;
  cout<<"(4.*transpose(B))*(0.5*C): "<<((4.*transpose(B))*(0.5*C))<<endl;
  
  cout<<"C*transpose(A): "<<(C*transpose(A))<<endl;
  cout<<"(2.*C)*transpose(A): "<<((2.*C)*transpose(A))<<endl;
  cout<<"C*(2.*transpose(A)): "<<(C*(2.*transpose(A)))<<endl;
  cout<<"(4.*C)*(0.5*transpose(A)): "<<((4.*C)*(0.5*transpose(A)))<<endl;
  
  cout<<"transpose(B)*transpose(A): "<<(transpose(B)*transpose(A))<<endl;
  cout<<"(2.*transpose(B))*transpose(A): "<<((2.*transpose(B))*transpose(A))<<endl;
  cout<<"transpose(B)*(2.*transpose(A)): "<<(transpose(B)*(2.*transpose(A)))<<endl;
  cout<<"(4.*transpose(B))*(0.5*transpose(A)): "<<((4.*transpose(B))*(0.5*transpose(A)))<<endl;
  
  tensor_type<double> AA(m,m,m,m);
  for (size_t i=0; i<m; ++i)
    for (size_t j=0; j<m; ++j)
      for (size_t k=0; k<m; ++k)
        for (size_t l=0; l<m; ++l)
          AA[i][j][k][l] = i + m*j + m*m*k + m*m*m*l;
  
  cout<<"AA: "<<AA<<endl;
  
  return 0;
}  
