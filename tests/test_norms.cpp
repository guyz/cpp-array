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

/*! \file test_norms.cpp
 *
 * \brief This file tests the norms of algebraic objects.
 */

#include "array.hpp"

using std::cout;
using std::endl;

using array::Norm_1;
using array::Norm_2;
using array::Norm_inf;
using array::vector_type;
using array::matrix_type;

int main(int argc, char **argv) {

  array::vector_type<float> xs = { 1, 2, 3, 3, 4, 5, 6, 7, 9 };

  cout << "Vector (single precision):\n  " << xs << endl;
  cout << "Norm 1: " << xs.norm(Norm_1) << endl;
  cout << "Norm 2: " << xs.norm(Norm_2) << endl;
  cout << "Norm infinity: " << xs.norm(Norm_inf) << endl;

  array::vector_type<double> xd = { 1, 2, 3, 3, 4, 5, 6, 7, 9 };

  cout << "Vector (double precision):\n  " << xd << endl;
  cout << "Norm 1: " << xd.norm(Norm_1) << endl;
  cout << "Norm 2: " << xd.norm(Norm_2) << endl;
  cout << "Norm infinity: " << xd.norm(Norm_inf) << endl;

  array::vector_type<int> xi = { 1, 2, 3, 3, 4, 5, 6, 7, 9 };

  cout << "Vector (integers):\n  " << xi << endl;
  cout << "Norm 1: " << xi.norm(Norm_1) << endl;
  cout << "Norm 2: " << xi.norm(Norm_2) << endl;
  cout << "Norm infinity: " << xi.norm(Norm_inf) << endl;

  array::matrix_type<float> As = { { 1, 2, 3 }, { 3, 4, 5 }, { 6, 7, 9 } };

  cout << "Matrix (single precision):\n  " << As << endl;
  cout << "Norm 1: " << As.norm(Norm_1) << endl;
  cout << "Norm infinity: " << As.norm(Norm_inf) << endl;

  array::matrix_type<double> Ad = { { 1, 2, 3 }, { 3, 4, 5 }, { 6, 7, 9 } };

  cout << "Matrix (double precision):\n  " << Ad << endl;
  cout << "Norm 1: " << Ad.norm(Norm_1) << endl;
  cout << "Norm infinity: " << Ad.norm(Norm_inf) << endl;

  array::matrix_type<int> Ai = { { 1, 2, 3 }, { 3, 4, 5 }, { 6, 7, 9 } };

  cout << "Matrix (integers):\n  " << Ai << endl;
  cout << "Norm 1: " << Ai.norm(Norm_1) << endl;
  cout << "Norm infinity: " << Ai.norm(Norm_inf) << endl;

  return 0;
}
