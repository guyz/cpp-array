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


/* This function tests the creation of algebraic objects by wrapping
 * existing arrays, by using lambda expressions and initializer lists.
 */


#include <iostream>
#include <vector>

#include "array-config.hpp"

#ifndef ARRAY_VERBOSE
#define ARRAY_VERBOSE 1
#endif

#include "array.hpp"

using std::cout;
using std::endl;

using array::vector_type;
using array::matrix_type;
using array::tensor_type;

int main() {
  
  // constructors for wrapped objects
  
  // initialize vector
  std::vector<double> v(16,1.);
  for (size_t i=0; i<v.size(); ++i)
    v[i] = i;
  
  vector_type<double> x(16, &v[0]);
  cout<<"x -> "<<x<<endl;
  
  array::matrix_type<double> X(4, &v[0]);
  cout<<"X -> "<<X<<endl;
  
  array::tensor_type<double> XX(2, &v[0]);
  cout<<"XX -> "<<XX<<endl;

    
  // constructors for lambda expressions
  
  int m = 5;
  
  vector_type<double> a(m, [=](int i) {return i % 2 == 0 ? 1 : 0;});
  cout<<"a -> "<<a<<endl;
  
  matrix_type<double> A(m, [=](int i, int j) {return i == j ? 1 : 0;});
  cout<<"A -> "<<A<<endl;
  
  matrix_type<double> B(m, [=](int i, int j) {return i*j % 2 == 0 ? 1 : 0;});
  cout<<"B -> "<<B<<endl;
  
  tensor_type<double> II(3, [=](int i, int j, int k, int l) { if (i == j) if (j == k) if (k == l) return 1; return 0;});
  cout<<"II -> "<<II<<endl;
  
  
  // constructors for initializer lists
  
  vector_type<double> y = { 1., 2., 3., 4.};
  cout<<"y -> "<<y<<endl;
  
  matrix_type<double> C = { {1., 2., 3.}, {4., 5., 6.} };
  cout<<"C -> "<<C<<endl;
  
  tensor_type<double> TT = { {{{1.}, {7.}, {13.}, {19}}, {{2}, {8}, {14}, {20}}, {{3}, {9}, {15}, {21}}}, {{{4.}, {10}, {16}, {22}}, {{5}, {11}, {17}, {23}}, {{6}, {12}, {18}, {24}}} };
  cout<<"TT -> "<<TT<<endl;
  
  return 0;
}