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

/*! \file return_type.hpp
 *
 * \brief This file contains forward declarations and the Return_type helper
 * structure that is used to discern the return type of an expression.
 */

#ifndef ARRAY_RETURN_TYPE_HPP
#define ARRAY_RETURN_TYPE_HPP

#include "array_fwd.hpp"

__BEGIN_ARRAY_NAMESPACE__

//! Return type class template, declared but never defined (see the partial
// template specializatoins).
template <typename... Params> struct Return_type;

//! Return type for an arbitrary binary expression with arbitrary operation
template <typename A, typename B, class Op>
struct Return_type<Expr<BinExprOp<A, B, Op> > > {
  typedef typename Return_type<A>::result_type left_result;
  typedef typename Return_type<B>::result_type right_result;
  typedef typename Return_type<left_result, right_result, Op>::result_type
  result_type;
};

//! Return type for the operation of an arbitrary object with an arbitrary
// expression
template <typename A, typename B, class Op> struct Return_type<A, Expr<B>, Op> {
  typedef typename Return_type<typename B::left_type, typename B::right_type,
                               typename B::operator_type>::result_type
  right_result;
  typedef typename Return_type<A, right_result, Op>::result_type result_type;
};

//! Return type for the operation of an arbitrary expression with an arbitrary
// object
template <typename A, typename B, class Op> struct Return_type<Expr<A>, B, Op> {
  typedef typename Return_type<typename A::left_type, typename A::right_type,
                               typename A::operator_type>::result_type
  left_result;
  typedef typename Return_type<left_result, B, Op>::result_type result_type;
};

//! Return type for the operation between two arbitrary expressions
template <typename A, typename B, class Op>
struct Return_type<Expr<A>, Expr<B>, Op> {
  typedef typename Return_type<typename A::left_type, typename A::right_type,
                               typename A::operator_type>::result_type
  left_result;
  typedef typename Return_type<typename B::left_type, typename B::right_type,
                               typename B::operator_type>::result_type
  right_result;
  typedef typename Return_type<left_result, right_result, Op>::result_type
  result_type;
};

// partial template specializations for individual cases

//! Return type between scalar operations
template <typename S, class Op>
struct Return_type<typename std::enable_if<std::is_arithmetic<S>::value, S>::type, S, Op> {
  typedef S result_type;
};

//! Return type for scalar - array operations
template <int d, typename S, typename T, class Op>
struct Return_type<ExprLiteral<S>, Array<d, T>, Op> {
  typedef Array<d, T> result_type;
};

//! Return type for vector transposition
template <typename T> struct Return_type<Array<1, T>, EmptyType, ApTr> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr> result_type;
};

//! Return type for matrix transposition
template <int d, typename T> struct Return_type<Array<d, T>, EmptyType, ApTr> {
  typedef Array<d, T> result_type;
};

//! Return type for transposed vector - vector multiplication
template <typename T>
struct Return_type<BinExprOp<Array<1, T>, EmptyType, ApTr>, Array<1, T>,
                   ApMul> {
  typedef T result_type;
};

//! Return type for vector - transposed vector multiplication
template <typename T>
struct Return_type<Array<1, T>, BinExprOp<Array<1, T>, EmptyType, ApTr>,
                   ApMul> {
  typedef Array<2, T> result_type;
};

//! Return type for matrix - transposed vector multiplication
template <int d, typename T>
struct Return_type<Array<d, T>, BinExprOp<Array<1, T>, EmptyType, ApTr>,
ApMul> {
  typedef Array<d, T> result_type;
};

//! Return type for scalar - transposed vector multiplication
template <typename T>
struct Return_type<ExprLiteral<T>, BinExprOp<Array<1, T>, EmptyType, ApTr>,
                   ApMul> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr> result_type;
};

//! Return type for matrix - vector multiplication
template <typename T> struct Return_type<Array<2, T>, Array<1, T>, ApMul> {
  typedef Array<1, T> result_type;
};

//! Return type for vector - matrix multiplication
template <typename T> struct Return_type<Array<1, T>, Array<2, T>, ApMul> {
  typedef Array<2, T> result_type;
};

//! Return type for operations between matrices of the same dimension
template <int d, typename T, class Op>
struct Return_type<Array<d, T>, Array<d, T>, Op> {
  typedef Array<d, T> result_type;
};

//! Return type for transposed vector - matrix multiplication
template <typename T>
struct Return_type<BinExprOp<Array<1, T>, EmptyType, ApTr>, Array<2, T>,
                   ApMul> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr> result_type;
};

__END_ARRAY_NAMESPACE__

#endif /* ARRAY_RETURN_TYPE_HPP */
