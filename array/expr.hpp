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

/*! \file expr.hpp
 *
 * \brief This files contains the expression template machinery used to provide
 * the users with mahtematical notation without compromising performance.
 */

#ifndef ARRAY_EXPR_HPP
#define ARRAY_EXPR_HPP

#include <iostream>
#include <typeinfo>
#include <utility>
#include <type_traits>

#include "array-config.hpp"
#include "array.hpp"
#include "blas.hpp"


__BEGIN_ARRAY_NAMESPACE__


using std::cout;
using std::endl;
using std::is_arithmetic;


//! Expression identity, placeholder for a variable
template <typename T>
class ExprIdentity {
  
public:
  
  typedef T value_type;
  typedef T result_type;
  
  value_type operator()(T x) const
  { return x; }
};


//! Expression literal, represents a value that appears in the expression
template <typename T>
class ExprLiteral {
  
public:
  
  typedef T value_type;
  
  ExprLiteral(value_type value) : value_(value) {}
  
  template <typename... Args>
  value_type operator()(Args... params) const {
    return value_;
  }
  
  // provide implicit conversion
  operator value_type() const
  { return value_; }
  
private:
  value_type value_;
};


//! Expression wrapper class
template <class A>
class Expr {
  
private:
  A a_;
  
public:
  
  //  typedef typename A::value_type value_type;
  typedef typename A::result_type result_type;
  typedef A expression_type;
  
  const A& expr() const
  { return a_; }
  
  auto left() const -> decltype(a_.left())  { return a_.left(); }
  auto right() const -> decltype(a_.right()) { return a_.right(); }
  
  Expr() : a_() {}
  
  Expr(const A& x) : a_(x) {}
    
  operator result_type()
  { return a_(); }
  
  result_type operator()() const
  { return a_(); }
  
  result_type operator()(double x) const
  { return a_(x); }
  
  friend inline std::ostream& operator<<(std::ostream& os, const Expr<A>& expr) {
    return print<A>(os,expr);
  }
  
};


//! Expression traits class template, never defined.
template <class>
struct Expr_traits;

//! Expression traits partial template specialization for any expression.
template <class A>
struct Expr_traits<Expr<A> > {
  typedef Expr<A> type;
};

//! Expression traits partial template specialization for a literal.
template <typename T>
struct Expr_traits<ExprLiteral<T> > {
  typedef ExprLiteral<T> type;
};

//! Expression traits partial template specialization for an array.
template <int d, typename T>
struct Expr_traits<Array<d,T> > {
  typedef const Array<d,T>& type;
};

//! Expression traits partial template specialization for the empty type structure.
template <>
struct Expr_traits<EmptyType > {
  typedef EmptyType type;
};



//! Wrapper on a binary expression
/*! \tparam A - Left branch of the expression
 * \tparam B - Right branch of the expression
 * \tparam Op - Operator to be applied to between the branches
 */
template<class A, class B, class Op>
class BinExprOp {
  
  typename Expr_traits<A>::type a_;
  typename Expr_traits<B>::type b_;
  
public:
  
  typedef A left_type;
  typedef B right_type;
  typedef Op operator_type;
  typedef typename Return_type<left_type, right_type, operator_type>::result_type result_type;
  
  //! Parameter constructor
  BinExprOp(const A& a, const B& b) : a_(a), b_(b) {}

  //! Left branch
  auto left() const -> decltype(a_)
  { return a_; }
  
  //! Right branch
  auto right() const -> decltype(b_)
  { return b_; }
  
  //! Overloaded operator() for evaluating the binary expression
  auto operator()() const -> decltype(Op::apply(a_, b_))
  { return Op::apply(a_, b_); }
  
  //! Overloaded operator() for evaluating the binary expression
  auto operator()(double x) const -> decltype(Op::apply(a_,b_))
  { return Op::apply(a_(x), b_(x)); }
};


//! Wrapper on a binary expression used with lvalue references
/*! \tparam A - Reference to the left branch of the expression
 * \tparam B - Right branch of the expression
 * \tparam Op - Operator to be applied to between the branches
 */
template<class A, class B, class Op>
class RefBinExprOp {
  
public:
  
  typedef A& reference_type;
  
private:
  
  reference_type a_;
  B b_;
  
public:
  
  typedef reference_type left_type;
  typedef B right_type;
  typedef reference_type result_type;

  //! Parameter constructor
  RefBinExprOp(reference_type a, const B& b)
  : a_(a), b_(b) {}
  
  //! Left branch
  left_type left() const
  { return a_; }
  
  //! Right branch
  const right_type& right() const
  { return b_; }
  
  //! Overloaded operator() for evaluating the binary expression
  reference_type operator()() const
  { return Op::apply(a_, b_); }
  
  //! Overloaded operator() for evaluating the binary expression
  reference_type operator()(double x) const
  { return Op::apply(a_(x), b_(x)); }
  
  friend std::ostream& operator<<(std::ostream& os, const RefBinExprOp& bop)
  { return os; }
};


////////////////////////////////////////////////////////////////////////////////
// alias templates

//! scalar -- array multiplication
template <int d, typename T>
using SAm = Expr<BinExprOp< ExprLiteral<T>, Array<d,T>, ApMul> >;

//! scalar -- vector multiplication
template <typename T>
using SVm = SAm<1,T>;

//! scalar -- matrix multiplication
template <typename T>
using SMm = SAm<2,T>;


//! expression -- (scalar -- array multiplication) multiplication alias template
template <int d, typename T, typename A>
using ESAmm = Expr< BinExprOp< Expr<A>, SAm<d,T>, ApMul> >;


//! (scalar -- matrix multiplication) -- (scalar -- matrix multiplication) multiplication
template <typename T>
using SMmSMmm = Expr< BinExprOp< SMm<T>, SMm<T>, ApMul > >;

//! vector transposition
template <typename T>
using Vt = Expr<BinExprOp<vector_type<T>, EmptyType, ApTr> >;

//! array transposition
template <int d, typename T>
using At = Expr< BinExprOp<Array<d,T>, EmptyType, ApTr> >;


//! scalar*transposed vector multiplication
template <typename T>
using SVtm = Expr<BinExprOp<ExprLiteral<T>, Vt<T>, ApMul> >;

//! scalar*transposed matrix multiplication
template <typename T>
using SMtm = Expr< BinExprOp< ExprLiteral<T>, Expr< BinExprOp< matrix_type<T>, EmptyType, ApTr> >, ApMul> >;


//! scalar*transposed vector -- scalar matrix multiplication
template <typename T>
using SVtmSMmm = Expr<BinExprOp< SVtm<T>, SMm<T>, ApMul> >;



////////////////////////////////////////////////////////////////////////////////
// applicative classes



//! Applicative class for the addition operation
class ApAdd {
public:
  
  ////////////////////////////////////////////////////////////////////////////////
  // return references
  
  //! array -- scalar*array addition
  template <int d, typename T>
  static Array<d,T>& apply(Array<d,T>& a, const SAm<d,T>& y) {
    
    const Array<d,T>& b = y.right();
    
    // size assertion
    for (size_t i=0; i<d; ++i)
      assert(a.n_[i] == b.n_[i]);
    
    cblas_axpy<T>(a.size(), y.left(), b.data_, 1, a.data_, 1);
    return a;
  }
  
  //! array -- array addition
  template <int d, typename T>
  static Array<d,T> apply(const Array<d,T>& a, const Array<d,T>& b) {
    
    // size assertion
    for (size_t i=0; i<d; ++i)
      assert(a.n_[i] == b.n_[i]);
    
    Array<d,T> r(b);
    cblas_axpy<T>(a.size(), T(1), a.data_, 1, r.data_, 1);
    return r;
  }
  
  //! array -- (scalar*array -- scalar*array multiplication) addition
  template <typename T>
  static matrix_type<T>& apply(matrix_type<T>& c, const SMmSMmm<T>& y) {
    
    // get matrix refernces
    const matrix_type<T>& a = y.left().right();
    const matrix_type<T>& b = y.right().right();
    
    // check size
    assert(a.columns() == b.rows());
    assert(c.rows() == a.rows());
    assert(c.columns() == b.columns());
    
    cblas_gemm<T>(CblasNoTrans, CblasNoTrans, a.rows(), b.columns(),
               a.columns(), y.left().left()*y.right().left(),
               a.data_, a.rows(), b.data_, b.rows(), 1.0, c.data_, c.rows());
    return c;
  }
  
  //! array -- expr addition
  template<int d, typename T, class B>
  static Array<d,T>& apply(Array<d,T>& a, const Expr<B>& b) {
    return a += b();
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // return new objects
  
  //! scalar*array -- scalar*array addition
  template <int d, typename T>
  static Array<d,T> apply(const SAm<d,T>& x, const SAm<d,T>& y) {
    
    // get matrix refernces
    const Array<d,T>& a = x.right();
    const Array<d,T>& b = y.right();
    
    // size assertion
    for (size_t i=0; i<d; ++i)
      assert(a.n_[i] == b.n_[i]);
    
    // initialize result to first product
    T s = x.left();
    Array<d,T> r = s*a;
    
    // add second product
    cblas_axpy<T>(a.size(), y.left(), b.data_, 1, r.data_, 1);
    return r;
  }
  
  //! expr -- expr addition
  template<class A, class B>
  static typename Return_type<Expr<A>, Expr<B>, ApAdd>::result_type
  apply(const Expr<A>& a, const Expr<B>& b) {
    return a()+b();
  }
};

//! Applicative class for the subtraction operation
class ApSub {
public:
  
  //! array -- array addition
  template <int d, typename T>
  static Array<d,T> apply(const Array<d,T>& a, const Array<d,T>& b) {
    
    // size assertion
    for (size_t i=0; i<d; ++i)
      assert(a.n_[i] == b.n_[i]);
    
    Array<d,T> r(a);
    cblas_axpy<T>(r.size(), T(-1), b.data_, 1, r.data_, 1);
    return r;
  }
  
  //! expr -- expr addition
  template<class A, class B>
  static typename Return_type<Expr<A>, Expr<B>, ApSub>::result_type
  apply(const Expr<A>& a, const Expr<B>& b) {
    return a()-b();
  }
};

//! Applicative class for the multiplication operation
class ApMul {
public:
  
  //! scalar types
  template <typename T>
  static ExprLiteral<T> apply(const ExprLiteral<T>& a, const ExprLiteral<T>& b)
  { return ExprLiteral<T>(a*b); }
  
  //! scalar -- array multiplication
  template <int d, typename T>
  static Array<d,T> apply(const ExprLiteral<T>& a, const Array<d,T>& b) {
    
    // \todo this could be replaced by combining the constructor and
    // initialization using the scalar
    Array<d,T> r(b);
    cblas_scal<T>(b.size(), a, r.data_, 1);
    return r;
  }
  
  //! scalar -- transposed array multiplication
  template <int d, typename T>
  static Array<d,T> apply(const ExprLiteral<T>& a, const At<d,T>& b) {
    T s = a;
    return s * b();
  }
  
  //! transposed vector -- scalar*vector multiplication
  template <typename T>
  static T apply(const Vt<T>& a, const SVm<T>& b) {
    
    const vector_type<T>& x = a.left();
    const vector_type<T>& y = b.right();
    
    assert(x.size() == y.size());
    return b.left()*cblas_dot(x.size(), x.data_, 1, y.data_, 1);
  }
  
  //! scalar*vector -- transposed vector multiplication
  template <typename T>
  static matrix_type<T> apply(const SVm<T>& a, const Vt<T>& b) {
    
    const vector_type<T>& x = a.right();
    const vector_type<T>& y = b.left();
    
    matrix_type<T> r(x.size(), y.size());
    
    cblas_ger<T>(x.size(), y.size(), a.left(), x.data_, 1, y.data_, 1, r.data_, r.rows());
    return r;
  }
  
  //! scalar*transposed vector -- scalar*vector multiplication
  template <typename T>
  static ExprLiteral<T> apply(const SVtm<T>& a, const SVm<T>& b) {
    
    const vector_type<T>& x = a.right().left();
    const vector_type<T>& y = b.right();
    
    assert(x.size() == y.size());
    T dot = cblas_dot<T>(x.size(), x.data_, 1, y.data_, 1);
    
    return ExprLiteral<T>((b.left())*a.left()*dot);
  }
  
  //! scalar*vector -- scalar*transposed vector multiplication
  template <typename T>
  static matrix_type<T> apply(const SVm<T>& a, const SVtm<T>& b) {
    
    const vector_type<T>& x = a.right();
    const vector_type<T>& y = b.right().left();
    
    matrix_type<T> r(x.size(), y.size());
    
    cblas_ger<T>(x.size(), y.size(), a.left()*b.left(), x.data_, 1, y.data_, 1, r.data_, r.n_[0]);
    return r;
  }
  
  //! (scalar -- matrix multiplication) -- (scalar -- matrix multiplication) multiplication
  template <typename T>
  static matrix_type<T> apply(const SMm<T>& x, const SMm<T>& y) {
    
    // get matrix refernces
    const matrix_type<T>& a = x.right();
    const matrix_type<T>& b = y.right();
    
    // check size
    assert(a.columns() == b.rows());
    
    matrix_type<T> r(a.rows(), b.columns());
    cblas_gemm<T>(CblasNoTrans, CblasNoTrans, r.rows(), r.columns(),
               a.columns(), x.left()*y.left(),
               a.data_, a.rows(), b.data_, b.rows(), 0.0, r.data_, r.rows());
    return r;
  }
  
  //! scalar*matrix -- scalar*vector multiplication
  template <typename T>
  static vector_type<T> apply(const SMm<T>& x, const SVm<T>& y) {
    
    // get matrix refernces
    const matrix_type<T>& a = x.right();
    const vector_type<T>& b = y.right();
    
    // check size
    assert(a.columns() == b.size());
    
    vector_type<T> r(a.rows());
    cblas_gemv<T>(CblasNoTrans, a.rows(), a.columns(), x.left() * y.left(),
               a.data_, a.rows(), b.data_, 1, 0., r.data_, 1);
    return r;
  }
  
  //! scalar*transposed matrix -- scalar*matrix multiplication
  template <typename T>
  static matrix_type<T> apply(const SMtm<T>&x, const SMm<T> &y) {
    
    // get matrix refernces
    const matrix_type<T>& a = x.right().left();
    const matrix_type<T>& b = y.right();
    
    // check size
    assert(a.rows() == b.rows());
        
    matrix_type<T> r(a.columns(), b.columns());
    cblas_gemm<T>(CblasTrans, CblasNoTrans, r.rows(), r.columns(),
               a.rows(), x.left()*y.left(),
               a.data_, a.rows(), b.data_, b.rows(), 1.0, r.data_, r.rows());
    return r;
  }
  
  
  //! scalar*matrix -- scalar*transposed matrix multiplication
  template <typename T>
  static matrix_type<T> apply(const SMm<T> &x, const SMtm<T> &y) {
    
    // get matrix refernces
    const matrix_type<T>& a = x.right();
    const matrix_type<T>& b = y.right().left();
    
    // check size
    assert(a.columns() == b.columns());
    
    matrix_type<T> r(a.rows(), b.rows());
    cblas_gemm(CblasNoTrans, CblasTrans, r.rows(), r.columns(),
               a.columns(), x.left()*y.left(),
               a.data_, a.rows(), b.data_, b.rows(), 1.0, r.data_, r.rows());
    return r;
  }
  
  //! scalar*transposed matrix -- scalar*transposed matrix multiplication
  template <typename T>
  static matrix_type<T> apply(const SMtm<T> &x, const SMtm<T> &y) {
    
    // get matrix refernces
    const matrix_type<T>& a = x.right().left();
    const matrix_type<T>& b = y.right().left();
    
    // check size
    assert(a.rows() == b.columns());
    
    matrix_type<T> r(a.columns(), b.rows());
    cblas_gemm(CblasTrans, CblasTrans, r.rows(), r.columns(),
               a.rows(), x.left()*y.left(),
               a.data_, a.rows(), b.data_, b.rows(), 1.0, r.data_, r.rows());
    return r;
  }
  
  //! transposed vector -- expr multiplication
  template<typename T, class B>
  static typename Return_type<SVtm<T>, Expr<B>, ApMul>::result_type
  apply(const SVtm<T>& a, const Expr<B>& b) {
    return a*b();
  }
  
  
  //! transposed vector -- matrix multiplication -- expression multiplication
  template<typename T, class B>
  static typename Return_type<SVtm<T>, Expr<B>, ApMul>::result_type
  apply(const SVtmSMmm<T>& a, const Expr<B>& b) {
    
    T s = a.left().left() * a.right().left();
    const vector_type<T>& v = a.left().right().left();
    const matrix_type<T>& m = a.right().right();
    
    assert(v.size() == m.rows());
    vector_type<T> r(m.columns());
    
    cblas_gemm(CblasNoTrans,CblasNoTrans, 1, m.columns(),
               v.size(), s, v.data_, 1, m.data_, m.rows(),
               1.0, r.data_, 1);
    
    return transpose(r)*b;
  }
  
  //! expr -- expr multiplication
  template<class A, class B>
  static typename Return_type<Expr<A>, Expr<B>, ApMul>::result_type
  apply(const Expr<A>& a, const Expr<B>& b) {
    return a()*b();
  }
};

//! Applicative class for the division operation
class ApDiv {
public:
  
  
  // expr -- expr divition
  template<class A, class B>
  static typename Return_type<Expr<A>, Expr<B>, ApDiv>::result_type
  apply(const Expr<A>& a, const Expr<B>& b) {
    return a()/b();
  }
};


//! ApTr -- transpose
class ApTr {
  
  public:
  
  // this function should never be called
  template <typename T>
  static inline vector_type<T> apply(const vector_type<T>& a, EmptyType) {
    cout<<"*** ERROR *** Cannot return the transpose of a vector"<<endl;
    exit(1);
  }
  
  //! Transpose matrix
  template <typename T>
  static inline matrix_type<T> apply(const matrix_type<T>& a, EmptyType) {
    
    matrix_type<T> r(a.columns(), a.rows());
    for(size_t i=0; i<r.rows(); ++i)
      for(size_t j=0; j<r.columns(); ++j)
        r(i,j) = a(j,i);
    return r;
  }
};


////////////////////////////////////////////////////////////////////////////////
// overload operators


////////////////////////////////////////////////////////////////////////////////
// unary operator+

//! unary operator+(any)
template <class A>
typename std::enable_if<!is_arithmetic<A>::value, A>::type
operator+(const A& a)
{ return a; }

////////////////////////////////////////////////////////////////////////////////
// unary operator-

//! unary operator-(any)
template <class A>
typename std::enable_if<!is_arithmetic<A>::value, Expr<BinExprOp<ExprLiteral<typename A::value_type>, A, ApMul> > >::type
operator-(const A& a) {
  
  typedef typename A::value_type value_type;
  return value_type(-1) * a;
}

////////////////////////////////////////////////////////////////////////////////
// operator+

//! operator+(expr, expr)
template<class A, class B>
Expr<BinExprOp<Expr<A>, Expr<B>, ApAdd> >
operator+(const Expr<A>& a, const Expr<B>& b) {
  
  typedef BinExprOp<Expr<A>, Expr<B>, ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}

//! operator+(array, array)
template <int d, typename T>
Expr<BinExprOp<Array<d,T>, Array<d,T>, ApAdd> >
operator+(const Array<d,T>& a, const Array<d,T>& b)
{
  
  typedef BinExprOp<Array<d,T>, Array<d,T>, ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}


//! operator+(expr, array)
template<int d, typename T, class A>
Expr< BinExprOp<
Expr<A>,
SAm<d,T>,
ApAdd> >
operator+(const Expr<A>& a, const Array<d,T>& b) {
  
  typedef BinExprOp<
  Expr<A>,
  SAm<d,T>,
  ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a, T(1)*b));
}

//! operator+(array, expr)
template<int d, typename T, class B>
Expr< BinExprOp<
SAm<d,T>,
Expr<B>,
ApAdd> >
operator+(const Array<d,T>& a, const Expr<B>& b) {
  
  typedef BinExprOp<
  SAm<d,T>,
  Expr<B>,
  ApAdd> ExprT;
  return Expr<ExprT>(ExprT(T(1)*a, b));
}


////////////////////////////////////////////////////////////////////////////////
// operator-

//! operator-(expr, expr)
template<class A, class B>
Expr<BinExprOp<Expr<A>, Expr<B>, ApSub> >
operator-(const Expr<A>& a, const Expr<B>& b) {
  
  typedef BinExprOp<Expr<A>, Expr<B>, ApSub> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}

//! operator-(array, array)
template <int d, typename T>
Expr<BinExprOp<Array<d,T>, Array<d,T>, ApSub> >
operator-(const Array<d,T>& a, const Array<d,T>& b) {
  
  typedef BinExprOp<Array<d,T>, Array<d,T>, ApSub> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}

//! operator-(expr, array)
template<int d, typename T, class A>
Expr< BinExprOp<
Expr<A>,
SAm<d,T>,
ApAdd> >
operator-(const Expr<A>& a, const Array<d,T>& b) {
  
  typedef BinExprOp<
  Expr<A>,
  SAm<d,T>,
  ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a, T(-1)*b));
}


//! operator-(array, expr)
template<int d, typename T, class B>
Expr<BinExprOp<Expr<BinExprOp<ExprLiteral<T>,Array<d,T>,ApMul> >,Expr<B>, ApAdd> >
operator-(const Array<d,T>& a, const Expr<B>& b) {
  return (a + (T(-1)*b));
}


//! operator-(array, scalar*array)
template <int d, class T>
Expr<
BinExprOp<
SAm<d,T>,
SAm<d,T>,
ApAdd>
>
operator-(const Array<d,T>& a, const SAm<d,T>& b) {
  
  typedef BinExprOp< SAm<d,T>, SAm<d,T>, ApAdd> ExprT;
  T factor = T(-1) * b.left();
  return Expr<ExprT>(ExprT(T(1)*a, factor*b.right()));
}


////////////////////////////////////////////////////////////////////////////////
// operator*

//! operator*(expr, expr)
template<class A, class B>
Expr<BinExprOp<Expr<A>, Expr<B>, ApMul> >
operator*(const Expr<A>& a, const Expr<B>& b) {
  
  typedef BinExprOp<Expr<A>, Expr<B>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}

//! operator*(array, array)
template<int d1, int d2, typename T>
Expr< BinExprOp<
Expr< BinExprOp< ExprLiteral<T>, Array<d1,T>, ApMul> >,
Expr< BinExprOp< ExprLiteral<T>, Array<d2,T>, ApMul> >,
ApMul> >
operator*(const Array<d1,T>& a, const Array<d2,T>& b) {
  
  typedef BinExprOp<
  Expr< BinExprOp< ExprLiteral<T>, Array<d1,T>, ApMul> >,
  Expr< BinExprOp< ExprLiteral<T>, Array<d2,T>, ApMul> >,
  ApMul> ExprT;
  return Expr<ExprT>(ExprT(T(1)*a, T(1)*b));
}

//! operator*(scalar, expr)
template <typename S, class B>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< ExprLiteral<typename Expr<B>::value_type >, Expr<B>, ApMul> > >::type
operator*(S a, const Expr<B>& b) {
  
  typedef typename Expr<B>::value_type value_type;
  typedef BinExprOp< ExprLiteral<value_type>, Expr<B>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<value_type>(a),b));
}

//! operator*(expr, scalar)
template <typename S, class B>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< ExprLiteral<typename Expr<B>::value_type >, Expr<B>, ApMul> > >::type
operator*(const Expr<B>& b, S a) {
  
  typedef typename Expr<B>::value_type value_type;
  typedef BinExprOp< ExprLiteral<value_type>, Expr<B>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<value_type>(a),b));
}

//! operator*(scalar, scalar*expr)
template <typename S, class T, class B>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< ExprLiteral<T>, Expr<B>, ApMul> > >::type
operator*(S a, const Expr<BinExprOp< ExprLiteral<T>, Expr<B>, ApMul> >& b) {
  
  typedef BinExprOp< ExprLiteral<T>, Expr<B>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<T>(a*b.left()),b.right()));
}


//! operator*(scalar, scalar*expr*expr)
template <typename S, class T, class A, class B>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> > >::type
operator*(S a, const Expr<BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> >& b) {
  
  typedef BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> ExprT;
  T scalar = a*b.left().left();
  const A& left_expr = b.left().right();
  const Expr<B>& right_expr = b.right();
  
  return Expr<ExprT>(ExprT(scalar*left_expr, right_expr));
}

//! operator*(scalar*expr*expr, scalar)
template <typename S, class T, class A, class B>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> > >::type
operator*(const Expr<BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> >& b, S a) {
  
  typedef BinExprOp< Expr<BinExprOp<ExprLiteral<T>, A , ApMul> >, Expr<B>, ApMul> ExprT;
  T scalar = a*b.left().left();
  const A& left_expr = b.left().right();
  const Expr<B>& right_expr = b.right();
  
  return Expr<ExprT>(ExprT(scalar*left_expr, right_expr));
}

//! operator*(scalar*expr, scalar)
template <typename S, class T, class A>
typename std::enable_if<is_arithmetic<S>::value, Expr<BinExprOp< ExprLiteral<T>, Expr<A>, ApMul> > >::type
operator*(const Expr<BinExprOp< ExprLiteral<T>, Expr<A>, ApMul> >& a, S b) {
  
  typedef BinExprOp< ExprLiteral<T>, Expr<A>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<T>(a.left()*b),a.right()));
}

//! operator*(scalar, array)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value, SAm<d,T> >::type
operator*(S a, const Array<d,T>& b) {
  
  typedef typename SAm<d,T>::expression_type ExprT;
  return SAm<d,T>(ExprT(ExprLiteral<T>(a),b));
}

//! operator*(array, scalar)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value, SAm<d,T> >::type
operator*(const Array<d,T>& a, S b) {
  
  typedef typename SAm<d,T>::expression_type ExprT;
  return SAm<d,T>(ExprT(ExprLiteral<T>(b),a));
}

//! operator*(scalar*array, scalar)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value, SAm<d,T> >::type
operator*(const SAm<d,T>& a, S b) {
  
  typedef typename SAm<d,T>::expression_type ExprT;
  return SAm<d,T>(ExprT(ExprLiteral<T>(a.left()*b),a.right()));
}

//! operator*(scalar, scalar*array)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value, SAm<d,T> >::type
operator*(S a, const SAm<d,T>& b) {
  
  typedef typename SAm<d,T>::expression_type ExprT;
  return SAm<d,T>(ExprT(ExprLiteral<T>(a*b.left()),b.right()));
}


//! operator*(expr, array)
template<int d, typename T, class A>
ESAmm<d,T,A>
operator*(const Expr<A>& a, const Array<d,T>& b) {
  
  typedef typename ESAmm<d,T,A>::expression_type ExprT;
  return ESAmm<d,T,A>(ExprT(a, T(1)*b));
}

//! operator*(array, expr)
template<int d, typename T, class B>
Expr< BinExprOp<
SAm<d,T>,
Expr<B>,
ApMul> >
operator*(const Array<d,T>& a, const Expr<B>& b) {
  
  typedef BinExprOp<
  SAm<d,T>,
  Expr<B>,
  ApMul> ExprT;
  return Expr<ExprT>(ExprT(T(1)*a, b));
}


//! operator*(array, scalar*expr)
template <int d, typename S, class T, class B>
Expr< BinExprOp<
SAm<d,T>,
Expr<B>,
ApMul> >
operator*(const Array<d,T>& a, const Expr<BinExprOp< ExprLiteral<T>, Expr<B>, ApMul> >& b) {
  
  typedef BinExprOp<
  ExprLiteral<T>,
  Expr<B>,
  ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<T>(a*b.left()),b.right()));
}


//! operator*(scalar, transposed object)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value,
Expr<
BinExprOp<
ExprLiteral<T>,
Expr<BinExprOp<Array<d,T>, EmptyType, ApTr> >,
ApMul>
>
>::type
operator*(S a, const Expr<BinExprOp<Array<d,T>, EmptyType, ApTr> >& b) {
  
  typedef BinExprOp< ExprLiteral<T>, Expr<BinExprOp<Array<d,T>, EmptyType, ApTr> >, ApMul> ExprT;
  return Expr<ExprT>(ExprT(ExprLiteral<T>(a),b));
}


////////////////////////////////////////////////////////////////////////////////
// operator/

//! operator/(expr, expr)
template<class A, class B>
Expr<BinExprOp<Expr<A>, Expr<B>, ApDiv> >
operator/(const Expr<A>& a, const Expr<B>& b) {
  
  typedef BinExprOp<Expr<A>, Expr<B>, ApDiv> ExprT;
  return Expr<ExprT>(ExprT(a,b));
}


//! operator/(array, scalar)
template <int d, typename S, typename T>
typename std::enable_if<is_arithmetic<S>::value, SAm<d,T> >::type
operator/(const Array<d,T>& a, S b) {
  
  typedef typename SAm<d,T>::expression_type ExprT;
  return SAm<d,T>(ExprT(ExprLiteral<T>(1/b),a));
}


////////////////////////////////////////////////////////////////////////////////
// operator transpose

template <class A>
Expr< BinExprOp< ExprLiteral<typename A::value_type>, Expr<BinExprOp<A, EmptyType, ApTr> >, ApMul> >
transpose(const A& a) {
  
  typedef typename A::value_type value_type;
  typedef BinExprOp<A, EmptyType, ApTr> TrExprT;
  typedef BinExprOp< ExprLiteral<value_type>, Expr< TrExprT>, ApMul> ExprT;
  return Expr<ExprT>(ExprT(value_type(1), Expr<TrExprT>(TrExprT(a, EmptyType()))));
}

template <class A>
Expr< BinExprOp< ExprLiteral<typename A::value_type>, A, ApMul> >
transpose(const Expr< BinExprOp< ExprLiteral<typename A::value_type>, Expr<BinExprOp<A, EmptyType, ApTr> >, ApMul> >& a) {
  
  typedef typename A::value_type value_type;
  typedef BinExprOp< ExprLiteral<value_type>, A, ApMul> ExprT;
  
  return Expr<ExprT>(ExprT(a.left(), a.right().left()));
}

template <class A>
Expr <BinExprOp< ExprLiteral<typename A::value_type>, A , ApMul> >
transpose(const Expr< BinExprOp< ExprLiteral<typename A::value_type>, Expr< BinExprOp< Expr< BinExprOp< ExprLiteral<typename A::value_type>, A , ApMul> >, EmptyType, ApTr> >, ApMul> > & a) {
  
  typedef typename A::value_type value_type;
  
  typedef BinExprOp< ExprLiteral<typename A::value_type>, A , ApMul> ExprT;
  value_type s = a.left() * a.right().left().left();
  return Expr<ExprT>(ExprT(s, a.right().left().right()));
}


////////////////////////////////////////////////////////////////////////////////
// operator+=


//! operator+=(array, array)
template <int d, typename T>
Array<d,T>&
operator+=(Array<d,T>& a, const Array<d,T>& b) {
  return a += T(1)*b;
}

//! operator+=(any, any)
template <class A, class B>
typename std::enable_if<!is_arithmetic<A>::value && !is_arithmetic<B>::value, A& >::type
operator+=(A& a, const B& b) {
  typedef RefBinExprOp<A, B, ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a,b))();
}

////////////////////////////////////////////////////////////////////////////////
// operator-=


//! operator-=(array, array)
template <int d, typename T>
Array<d,T>&
operator-=(Array<d,T>& a, const Array<d,T>& b) {
  return a += T(-1)*b;
}

//! operator-=(any, any)
template <class A, class B>
A& operator-=(A& a, const B& b) {
  return a += -1*b;
}



////////////////////////////////////////////////////////////////////////////////
// operator<<


//! standard output
template <class A>
inline std::ostream& print(std::ostream& os, const Expr<A>& e) {
  os<<e();
  return os;
}


__END_ARRAY_NAMESPACE__


#endif /* ARRAY_EXPR_HPP */
