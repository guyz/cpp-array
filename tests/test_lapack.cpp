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

/*! \file test_lapack.cpp
 *
 * \brief This file tests lapack support.
 */

#include "array.hpp"

using std::cout;
using std::endl;

int main(int argc, char **argv) {

  // inverse of a matrix

  array::matrix_type<double> A = { { 1, 2, 3 }, { 4, 5, 4 }, { 3, 2, 1 } };

  cout << "List constructed matrix {{1,2,3},{4,5,4},{3,2,1}}:\n" << A << endl;
  cout << "Inverse matrix\n:  " << inverse(A) << endl;

  
  // inverse of a singular matrix
  
  array::matrix_type<double> B = { { 1, 2}, { 2, 4 }};

  cout << "List constructed singular matrix {{1,2},{2,4}}:\n" << B << endl;

  
  try {
    cout << "Inverting matrix B"<<endl;
    inverse(B);
    
  } catch(std::exception& e) {
    cout<<e.what()<<endl;
  }
  
  
  return 0;
}
