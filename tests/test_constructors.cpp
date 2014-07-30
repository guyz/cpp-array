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

/*! \file test_constructors.cpp
 *
 * \brief This function tests all constructors and move semantics.
 */

#include <vector>

#include "array.hpp"

using std::cout;
using std::endl;

typedef array::vector_type<double> vector_type;
typedef array::matrix_type<double> matrix_type;
typedef array::tensor_type<double> tensor_type;

matrix_type create() {
  return { { 6., 5., 4. }, { 3., 2., 1. } };
}

int main() {

  // constructors for wrapped objects

  // initialize vector
  std::vector<double> v(16, 1.);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = i;

  vector_type x(16, &v[0]);
  cout << "Wrapped vector x:\n  " << x << endl;

  matrix_type X(4, &v[0]);
  cout << "Wrapped matrix X:\n  " << X << endl;

  tensor_type XX(2, &v[0]);
  cout << "Wrapped tensor XX:\n  " << XX << endl;

  // constructors for lambda expressions

  int m = 5;

  vector_type a(m, [=](int i) { return i % 2 == 0 ? 1 : 0; });
  cout << "Intercalated vector (lambda i: i % 2 == 0):\n  a -> " << a << endl;

  matrix_type A(m, [=](int i, int j) { return i == j ? 1 : 0; });
  cout << "Identity matrix (lambda i,j: i == j):\n  A -> " << A << endl;

  matrix_type B(m, [=](int i, int j) { return i * j % 2 == 0 ? 1 : 0; });
  cout << "Matrix (lambda i,j: i*j % 2 == 0):\n  B -> " << B << endl;

  tensor_type II(3, [=](int i, int j, int k,
                        int l) { return i == j && j == k && k == l; });
  cout << "Matrix (lambda i,j,k,l: i == j == k == l):\n  II -> " << II << endl;

  // constructors for initializer lists

  vector_type y = { 1., 2., 3., 4. };
  cout << "Initializer list vector {1,2,3,4}:\n  y -> " << y << endl;

  matrix_type C = { { 1., 2., 3. }, { 4., 5., 6. } };
  cout << "Initializer list matrix {{1,2,3},{4,5,6}}:\n  C -> " << C << endl;

  tensor_type TT = {
    { { { 1. }, { 7. }, { 13. }, { 19 } }, { { 2 }, { 8 }, { 14 }, { 20 } },
      { { 3 }, { 9 }, { 15 }, { 21 } } },
    { { { 4. }, { 10 }, { 16 }, { 22 } }, { { 5 }, { 11 }, { 17 }, { 23 } },
      { { 6 }, { 12 }, { 18 }, { 24 } } }
  };
  cout << "Initializer list tensor:\n  TT -> " << TT << endl;

  matrix_type D(2, 3);
  cout << "Default constructed matrix D:\n  " << D << endl;

  // copy constructor

  matrix_type E(2, 3, 1.);
  cout << "Parameter constructed matrix E:\n  " << E << endl;

  matrix_type F(E);
  cout << "Copy constructed matrix F:\n  " << F << endl;

  // assignment operator

  matrix_type G;
  G = C;
  cout << "Assigned matrix G=C:\n  " << G << endl;

  // move constructor

  matrix_type H(static_cast<matrix_type &&>(create()));
  cout << "Move constructed H:\n  " << H << endl;

  // pointer constructor

  v = std::vector<double>(4, 10.);

  matrix_type I(2, &v[0]);
  cout << "Pointer constructed matrix I:\n  " << I << endl;

  return 0;
}
