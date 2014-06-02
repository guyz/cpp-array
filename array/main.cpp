/*
 * Copyright (©) 2014 EPFL Alejandro M. Aragón
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

#include <iostream>
#include "expr.hpp"

using std::cout;
using std::endl;

int main(int argc, char **argv) {
  
  // create 10x10 identity matrix
  array::matrix_type<double> A(10, [=](int i, int j) {return i == j ? 1 : 0;});
  cout<<"A: "<<A<<endl;
  
  return 0;
}

