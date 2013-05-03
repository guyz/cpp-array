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


/* This function tests array iterators.
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

int main() {
  
  using array::matrix_type;

  size_t m = 5;
  matrix_type<double> A(m,m);
  
  int k=0;
  int l=-1;
  for (int i=0; i<m; ++i)
    for (int j=0; j<m; ++j)
      A(i,j) = ++k;
  
  cout<<" A "<<A<<endl;

  // iterator for non-constant objects
  
  for (matrix_type<double>::iterator it = A.begin(); it != A.end(); ++it)
    cout<<" "<<*it;
  cout<<endl;

  for (matrix_type<double>::reverse_iterator it = A.rbegin(); it != A.rend(); ++it)
    cout<<" "<<*it;
  cout<<endl;
  
  const matrix_type<double> C = A;

  // iterator for constant objects
  
  for (matrix_type<double>::const_iterator it = C.begin(); it != C.end(); ++it)
    cout<<" "<<*it;
  cout<<endl;
  
  for (matrix_type<double>::const_reverse_iterator it = C.rbegin(); it != C.rend(); ++it)
    cout<<" "<<*it;
  cout<<endl;
  
  
  // dimensional iterators
  
  cout<<"\nrow iteration"<<endl;
  for (matrix_type<double>::diterator<0> it = A.dbegin<0>(); it != A.dend<0>(); ++it)
    cout<<' '<<*it<<endl;
  
  cout<<"\ncolumn iteration"<<endl;
  for (matrix_type<double>::diterator<1> it = A.dbegin<1>(); it != A.dend<1>(); ++it)
    cout<<' '<<*it;
  cout<<"\n\n";
  
  k = 0;
  for (matrix_type<double>::diterator<0> it1 = A.dbegin<0>(); it1 != A.dend<0>(); ++it1) {
    cout<<"row "<<k++<<":";
    for (matrix_type<double>::diterator<1> it2 = A.dbegin<1>(it1); it2 != A.dend<1>(it1); ++it2)
      cout<<" "<<*it2;
    cout<<'\n';
  }
  
  // over constant object
  
  cout<<"\nrow iteration"<<endl;
  for (matrix_type<double>::diterator<0> it = C.dbegin<0>(); it != C.dend<0>(); ++it)
    cout<<' '<<*it<<endl;
  
  cout<<"\ncolumn iteration"<<endl;
  for (matrix_type<double>::diterator<1> it = C.dbegin<1>(); it != C.dend<1>(); ++it)
    cout<<' '<<*it;
  cout<<"\n\n";
  
  k = 0;
  for (matrix_type<double>::diterator<0> it1 = C.dbegin<0>(); it1 != C.dend<0>(); ++it1) {
    cout<<"row "<<k++<<":";
    for (matrix_type<double>::diterator<1> it2 = C.dbegin<1>(it1); it2 != C.dend<1>(it1); ++it2)
      cout<<" "<<*it2;
    cout<<'\n';
  }
  
  
  return 0;
}
