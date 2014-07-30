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

/*! \file test_functions.cpp
 *
 * \brief This file tests the functions in functions.hpp.
 */

#include "array.hpp"

using std::cout;
using std::endl;

int main(int argc, char **argv) {

  // identity tensors
  array::Array<2, double> I = array::identity<2>(3);
  cout << "Identity matrix I:\n  " << I << endl;

  array::Array<4, double> II = array::identity<4>(3);
  cout << "Identity tensor II:\n  " << II << endl;

  // vec function

  array::matrix_type<double> A = { { 1, 2, 3 }, { 4, 5, 4 }, { 3, 2, 1 } };
  cout << "List constructed matrix {{1,2},{3,4}}:\n" << A << endl;
  cout << "Stack columns into a vector (both member and non-member " << endl;
  cout << "vec functions can be used):\n  " << A.vec() << endl;

  return 0;
}
