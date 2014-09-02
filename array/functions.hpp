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

/*! \file functions.hpp
 *
 * \brief This file implements convenient functions when working with the
 * algebraic objects provided by the library.
 */

#ifndef CPPARRAY_FUNCTIONS_HPP
#define CPPARRAY_FUNCTIONS_HPP

#include <iostream>
#include <sstream>
#include "expr.hpp"

#include <stdexcept>


__BEGIN_ARRAY_NAMESPACE__

/*! Creates an identity tensor
 *
 * This functions creates a square tensor and puts the value 1 in the diagonal.
 */
template <int k, typename T> Array<k, T> identity(size_t s) {

  static_assert(k != 1, "Error: Cannot create identity vector");

  Array<k, T> a(s);
  size_t idx = 1;
  for (int i = 1; i < k; ++i)
    idx += pow(s, i);
  for (int i = 0; i < s; ++i)
    a.data_[i * idx] = 1.;
  return a;
}

/*! \brief Creates a vector from a matrix by stacking columns
 */
template <int k, typename T> Array<1, T> vec(const Array<k, T> &m) {
  return m.vec();
}

// Functioin templates used to cast Array objects

//! Cast scalar into an Array
template <class S, typename T>
typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<S, T>::value,
                        S>::type
algebraic_cast(T v) {
  S s(1);
  s.data()[0] = v;
  return s;
}

//! Cast Array into a scalar
template <class S, int k, typename T>
typename std::enable_if<std::is_arithmetic<S>::value, S>::type
algebraic_cast(const Array<k, T> &a) {
  return a.template algebraic_cast<S>();
}

//! Provide casting between arrays
template <class S, int k, typename T>
typename std::enable_if<!std::is_arithmetic<S>::value, S>::type
algebraic_cast(const Array<k, T> &m) {
  static_assert(
      S::rank() != Array<k, T>::rank(),
      "Error: Algebraic cast does not work for Array objects of the same type");
  return m.template algebraic_cast<S>();
}

#if defined(HAVE_LAPACK) || defined(HAVE_CLAPACK)

class SingularMatrixException : public std::runtime_error {

  size_t f_;

public:
  SingularMatrixException(size_t f)
      : std::runtime_error("Problem encountered factorizing matrix."), f_(f) {}

  virtual const char *what() const throw() {
    std::stringstream oss;
    oss << std::runtime_error::what() << "\nZero factor found in "
                                         "upper triangular matrix: u(" << f_
        << "," << f_ << ") = 0";

    return oss.str().c_str();
  }
};

template <int k, typename T> Array<k, T> inverse(const Array<k, T> &A) {

  static_assert(k == 2, "Error: Inverse can only be obtained for matrices");

  assert(A.rows() == A.columns());

  Array<k, T> i(A);

  int size = A.rows();
  int *IPIV = new int[size + 1];
  int LWORK = size * size;
  double *WORK = new T[LWORK];
  int INFO;

  lapack_getrf(size, size, i.data_, size, IPIV, &INFO);

  if (INFO != 0)
    throw SingularMatrixException(INFO);

  lapack_getri(size, i.data_, size, IPIV, WORK, LWORK, &INFO);

  delete IPIV;
  delete WORK;

  return i;
}

#endif /* HAVE_LAPACK */

__END_ARRAY_NAMESPACE__

#endif /* CPPARRAY_FUNCTIONS_HPP */
