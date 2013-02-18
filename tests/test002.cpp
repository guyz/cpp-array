/*
 * Copyright (C) 2013 by Alejandro M. Aragón
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

#include <iostream>

#include <vector>

#include "expr.hpp"

using std::cout;
using std::endl;

int main() {
  
  typedef array::vector_type<double> vector_type;
  typedef array::matrix_type<double> matrix_type;
  typedef array::tensor_type<double> tensor_type;

  // initialize vector
  std::vector<double> v(16,1.);
  for (size_t i=0; i<v.size(); ++i)
    v[i] = i;
  
  array::vector_type<double> x(&v[0], 16);
  cout<<"x -> "<<x<<endl;

  array::matrix_type<double> X(&v[0], 4);
  cout<<"X -> "<<X<<endl;

  array::tensor_type<double> XX(&v[0], 2);
  cout<<"XX -> "<<XX<<endl;

  return 0;
}
