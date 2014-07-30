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

/*! \file array_fwd.hpp
 *
 * \brief This file contains forward declarations.
 */

#ifndef ARRAY_FWD_HPP
#define ARRAY_FWD_HPP

#include "array-config.hpp"

__BEGIN_ARRAY_NAMESPACE__

enum Norm_type {
  Norm_1,
  Norm_2,
  Norm_inf
};

template <class output_stream>
output_stream &operator<<(output_stream &os, Norm_type n) {

  switch (n) {
  case Norm_1:
    os << "1-norm";
    break;
  case Norm_2:
    os << "2-norm";
    break;
  case Norm_inf:
    os << "Inf-norm";
    break;
  default:
    assert(false);
  }
  return os;
}

// class forward declarations
template <class> class Expr;

template <class A, class B, class Op> class BinExprOp;

template <int, typename> class Array;

template <class A> inline std::ostream &print(std::ostream &, const Expr<A> &);

template <typename T> class ExprLiteral;

template <typename T> class ExprIdentity;

class ApAdd;
class ApSub;
class ApMul;
class ApDiv;
class ApTr;

// function forward declarations

template <int k, typename T = double> Array<k, T> identity(size_t);

template <int k, typename T> Array<1, T> vec(const Array<k, T> &);

////! Functioin template used to cast Array objects
//template <typename casted_type, class k, typename T = double>
//casted_type algebraic_cast(const Array<k,T> &);

#if defined(HAVE_LAPACK) || defined(HAVE_CLAPACK)

template <int k, typename T = double> Array<k, T> inverse(const Array<k, T> &);

#endif /* HAVE_LAPACK */

//! Empty helper structure
struct EmptyType {
  typedef void value_type;
};

__END_ARRAY_NAMESPACE__

#endif /* ARRAY_FWD_HPP */
