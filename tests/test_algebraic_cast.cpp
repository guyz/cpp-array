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

/*! \file test_algebraic_cast.cpp
 *
 * \brief This function tests the casting between different objects.
 */

#include <iostream>
#include <iomanip>

#include "array.hpp"

using std::cout;
using std::endl;

using namespace array;

void separator() { cout << std::setw(40) << std::setfill('_') << " " << endl; }

int main(int argc, char **argv) {
  
  typedef array::vector_type<double> vector_type;
  typedef array::matrix_type<double> matrix_type;
  typedef array::tensor_type<double> tensor_type;
  
  separator();
  
  vector_type x = { 1. };
  matrix_type A = { { 1. } };
  tensor_type TT = { { { { 1. } } } };
  
  cout << "Vector x (double precision):\n  " << x << endl;
  cout << "Matrix A (double precision):\n  " << A << endl;
  cout << "Tensor TT (double precision):\n  " << TT << endl;
  
  separator();
  
  double s = algebraic_cast<double>(x);
  
  cout << "Cast vector to scalar: " << s << endl;
  cout << "Cast matrix to scalar: " << algebraic_cast<double>(A) << endl;
  cout << "Cast tensor to scalar: " << algebraic_cast<double>(TT) << endl;
  
  separator();
  
  cout << "Cast back scalar to vector:\n  " <<
  algebraic_cast<vector_type>(s)
  << endl;
  cout << "Cast back scalar to matrix:\n  " <<
  algebraic_cast<matrix_type>(s)
  << endl;
  cout << "Cast back scalar to tensor:\n  " <<
  algebraic_cast<tensor_type>(s)
  << endl;
  
  separator();
  
  vector_type y = { 1., 2., 3. };
  matrix_type B = algebraic_cast<matrix_type>(y);
  tensor_type SS = algebraic_cast<tensor_type>(y);
  
  cout << "Vector y (double precision):\n  " << y << endl;
  cout << "Cast vector to matrix:\n  " << B << endl;
  cout << "Cast vector to tensor:\n  " << SS << endl;
  
  separator();
  
  cout << "Cast back matrix to vector:\n  " <<
  algebraic_cast<vector_type>(B)
  << endl;
  cout << "Cast back tensor to vector:\n  " <<
  algebraic_cast<vector_type>(SS)
  << endl;
  
  separator();
  
  matrix_type C = { { 1., 2., 3. }, { 4., 5., 6. } };
  cout << "Matrix C (double precision):\n  " << C << endl;
  
  tensor_type RR = algebraic_cast<tensor_type>(C);
  
  cout << "Cast matrix to tensor:\n  " << RR << endl;
  
  separator();
  
  cout << "Cast back tensor to matrix:\n  " << algebraic_cast<matrix_type>(RR)
  << endl;
  
  return 0;
}
