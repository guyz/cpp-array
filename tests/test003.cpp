/*
 * Copyright (C) 2013 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
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


/*! \file test003.cpp
 *
 * \brief This function tests all constructors and move semantics.
 */


#include <vector>


#include "array.hpp"

using std::cout;
using std::endl;

typedef array::Array<2, double> array_type;


array_type create_empty() {
  return array_type(2);
}


int main() {
  
  array_type a(2,3,1.);
  cout<<"a -> "<<a<<endl;
  
  array_type b(a);
  cout<<"b -> "<<b<<endl;
  
  array_type c;
  c = b;
  cout<<"c -> "<<c<<endl;
  
  array_type d(static_cast<array_type&&>(create_empty()));
  cout<<"d -> "<<d<<endl;
  
  c = create_empty();
  cout<<"moved c -> "<<c<<endl;
    
  std::vector<double> v(4,10.);
  
  array_type e(2, &v[0]);
  cout<<"e -> "<<e<<endl;
  
  return 0;
}
