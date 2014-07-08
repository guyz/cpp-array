/*
 * Copyright (©) 2014 Alejandro M. Aragón
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

/*! \file return_type.hpp
 *
 * \brief This file contains forward declarations and the Return_type helper
 * structure that is used to discern the return type of an expression.
 */


#ifndef ARRAY_RETURN_TYPE_HPP
#define ARRAY_RETURN_TYPE_HPP

#include "array-config.hpp"

__BEGIN_ARRAY_NAMESPACE__

// forward declarations
template <class>
class Expr;

template<class A, class B, class Op>
class BinExprOp;

template <int, typename>
class Array;

template <class A>
inline std::ostream& print(std::ostream&, const Expr<A>&);

template <typename T>
class ExprLiteral;

template <typename T>
class ExprIdentity;

class ApAdd;
class ApSub;
class ApMul;
class ApDiv;
class ApTr;

//! Empty helper structure
struct EmptyType {
  typedef void value_type;
};

//! Return type class template, declared but never defined (see the partial template specializatoins).
template <typename... Params>
struct Return_type;

//! Return type for an arbitrary binary expression with arbitrary operation
template <typename A, typename B, class Op>
struct Return_type < Expr < BinExprOp <A, B, Op > > > {
  typedef typename Return_type<A>::result_type left_result;
  typedef typename Return_type<B>::result_type right_result;
  typedef typename Return_type<left_result, right_result, Op >::result_type result_type;
};

//! Return type for the operation of an arbitrary object with an arbitrary expression
template <typename A, typename B, class Op>
struct Return_type < A, Expr < B >, Op > {
  typedef typename Return_type<typename B::left_type,
  typename B::right_type, typename B::operator_type>::result_type right_result;
  typedef typename Return_type<A, right_result, Op >::result_type result_type;
};

//! Return type for the operation of an arbitrary expression with an arbitrary object 
template <typename A, typename B, class Op>
struct Return_type < Expr<A>, B, Op > {
  typedef typename Return_type<typename A::left_type,
  typename A::right_type, typename A::operator_type>::result_type left_result;
  typedef typename Return_type<left_result, B, Op >::result_type result_type;
};

//! Return type for the operation between two arbitrary expressions 
template <typename A, typename B, class Op>
struct Return_type < Expr<A>, Expr<B>, Op > {
  typedef typename Return_type<typename A::left_type,
  typename A::right_type, typename A::operator_type>::result_type left_result;
  typedef typename Return_type<typename B::left_type,
  typename B::right_type, typename B::operator_type>::result_type right_result;
  typedef typename Return_type<left_result, right_result, Op >::result_type result_type;
};

// partial template specializations for individual cases

//! Return type for operations involving objects and literals
template <typename S, class Op>
struct Return_type<S, ExprLiteral<S>, Op> {
  typedef S result_type;
};

//! Return type for operations involving objects and literals
template <typename S, class Op>
struct Return_type<ExprLiteral<S>, S, Op> {
  typedef S result_type;
};

//! Return type for scalar - matrix operations
template <int d, typename T, class Op>
struct Return_type<ExprLiteral<T>, Array<d, T>, Op> {
  typedef Array<d, T> result_type;
};

//! Return type for vector transposition
template <typename T>
struct Return_type<Array<1,T>, EmptyType, ApTr> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr>  result_type;
};

//! Return type for matrix transposition
template <int d, typename T>
struct Return_type<Array<d,T>, EmptyType, ApTr> {
  typedef Array<d,T>  result_type;
};

//! Return type for transposed vector - vector multiplication
template <typename T>
struct Return_type<BinExprOp<Array<1,T>, EmptyType, ApTr>, Array<1,T>, ApMul> {
  typedef T result_type;
};

//! Return type for vector - transposed vector multiplication
template <typename T>
struct Return_type<Array<1,T>, BinExprOp<Array<1,T>, EmptyType, ApTr>, ApMul> {
  typedef Array<2,T> result_type;
};

//! Return type for scalar - transposed vector multiplication
template <typename T>
struct Return_type<ExprLiteral<T>, BinExprOp<Array<1,T>, EmptyType, ApTr>, ApMul> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr> result_type;
};


//! Return type for matrix - vector multiplication
template <typename T>
struct Return_type<Array<2,T>, Array<1,T>, ApMul> {
  typedef Array<1,T> result_type;
};

//! Return type for operations between matrices of the same dimension
template <int d, typename T, class Op>
struct Return_type<Array<d, T>, Array<d, T>, Op> {
  typedef Array<d, T> result_type;
};


//! Return type for transposed vector - matrix multiplication
template <typename T>
struct Return_type<BinExprOp<Array<1,T>, EmptyType, ApTr>, Array<2, T>, ApMul> {
  typedef BinExprOp<Array<1, T>, EmptyType, ApTr> result_type;
};



__END_ARRAY_NAMESPACE__

#endif /* ARRAY_RETURN_TYPE_HPP */
