/*
 * Copyright (C) 2011 by Alejandro M. Aragón
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

//! \file array.hpp
//
//  Created by Alejandro Aragón on 10/11/11.
//

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <iostream>
#include <cassert>
#include <iomanip>
#include <initializer_list>

#include "return_type.hpp"

#define ARRAY_VERBOSE 1

__BEGIN_ARRAY_NAMESPACE__

using std::cout;
using std::endl;


template <int d>
struct Print;

template <int d, class Array>
struct Array_proxy;


template <bool B, class T = void>
struct enable_if { typedef T type; };

template <class T>
struct enable_if<false, T> {};


template <bool B, class T = void>
struct disable_if { typedef T type; };

template <class T>
struct disable_if<true, T> {};

// int 2 type construct
template <int d>
struct Int2Type {
  enum { result = d };
};

// type 2 type construct
template <class T>
class Type2Type {
  typedef T OriginalType;
};




template <int d, typename T>
class Array_base {
  
public:
  typedef T value_type;
  
  Array_base() : n_(), data_(0) {}
  
  // move constructor
  Array_base(Array_base&& src) : data_(src.data_) {
    
#ifdef ARRAY_VERBOSE
    cout<<"inside Array_base(Array&&)"<<endl;
#endif
    for (int i=0; i<d; ++i)
      n_[i] = src.n_[i];
    src.data_ = NULL;
  }
  
protected:
  size_t n_[d];
  value_type* data_;
};

template <typename T>
class Array_base<1,T> {
  
public:
  typedef T value_type;
  
  Array_base() : n_(), data_(0) {}
  
  size_t size() const
  { return n_[0]; }
  
  value_type norm() const {
    assert(n_[0] > 0);
    return norm(Type2Type<value_type>());
  }
  
  //    // move constructor
  //    Array_base(Array_base&& src) : data_(src.data_) {
  //
  //#ifdef ARRAY_VERBOSE
  //        cout<<"inside Array_base(Array&&)"<<endl;
  //#endif
  //        n_[0] = src.n_[0];
  //        src.data_ = NULL;
  //    }
  
private:
  
  template <class U>
  inline U norm(Type2Type<U>) const {
    U norm = U();
    for (size_t i=0; i<size(); ++i)
      norm += pow(data_[i],2);
    norm = sqrt(norm);
    return norm;
  }
  
  // norm function
  inline double norm(Type2Type<double>) const {
    // call to blas routine
    return cblas_dnrm2(n_[0], data_, 1);
  }
  
public:
  //protected:
  size_t n_[1];
  value_type* data_;
};


template <typename T>
class Array_base<2,T> {
  
public:
  typedef T value_type;
  
  Array_base() : n_(), data_(0) {}
  
  size_t rows() const
  { return n_[0]; }
  
  size_t columns() const
  { return n_[1]; }
  
  
  //    // move constructor
  //    Array_base(Array_base&& src) : data_(src.data_) {
  //
  //#ifdef ARRAY_VERBOSE
  //        cout<<"inside Array_base(Array&&)"<<endl;
  //#endif
  //        for (int i=0; i<2; ++i)
  //            n_[i] = src.n_[i];
  //        src.data_ = NULL;
  //    }
  
  
protected:
  size_t n_[2];
  value_type* data_;
};

template <class A, class B>
struct SameClass {
  enum { result = false };
};

template <class A>
struct SameClass<A,A> {
  enum { result = true };
};


template <typename T, typename P>
class Array_iterator : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, P>
{
  typedef P pointer;
  
  pointer p;
public:
  Array_iterator(T* x) :p(x) {}
  Array_iterator(const Array_iterator& mit) : p(mit.p) {}
  
  // Allow iterator to const_iterator conversion
  template<typename iterator>
  Array_iterator(const Array_iterator<T,
                 typename enable_if<(SameClass<P, typename iterator::pointer>::result), P>::type>& i)
  : p(i.p) {}
  
  Array_iterator& operator++()
  { ++p; return *this; }
  
  Array_iterator operator++(int) {
    return Array_iterator(p++);
  }
  
  Array_iterator& operator--()
  { --p; return *this; }
  
  Array_iterator operator--(int)
  { return Array_iterator(p--); }
  
  bool operator==(const Array_iterator& rhs)
  {return p == rhs.p;}
  
  bool operator!=(const Array_iterator& rhs)
  {return p != rhs.p;}
  
  T& operator*()
  {return *p;}
};

template <int d, class Array>
struct Array_proxy_traits {
  
  typedef Array_proxy<d-1, Array> reference_type;
  typedef const Array_proxy<d-1, Array> value_type;
  
  static reference_type reference(Array& a, size_t i)
  { return reference_type(a,i); }
  
  static value_type value(Array& a, size_t i)
  { return value_type(a,i); }
};


template <class Array>
struct Array_proxy_traits<1,Array> {
  typedef typename Array::value_type primitive_type;
  typedef primitive_type& reference_type;
  typedef primitive_type const & value_type;
  
  static reference_type reference(Array& a, size_t i)
  { return a.data_[i]; }
  
  static value_type value(Array& a, size_t i)
  { return a.data_[i]; }
};






/*! \tparam n - Dimension of array
 */
template <int d, typename T>
struct Array : public Array_base <d,T>  {
  
  typedef T* pointer_type;
  typedef T& reference_type;
  typedef T value_type;
  
  typedef Array_iterator<value_type, pointer_type> iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  
  static const int d_ = d;
  
  typedef Array_base <d,T> base_type;
  
  using base_type::n_;
  using base_type::data_;
  
  // default constructor
  Array() : base_type() {
#ifdef ARRAY_VERBOSE
    cout<<"Inside constructor template <typename... Args> explicit Array(const Args&... args)"<<endl;
#endif
  }
  
  
  template <int dim>
  void init() {
    
#ifdef ARRAY_VERBOSE
    cout<<"Inside init(), initializing memory"<<dim<<endl;
#endif
    
    size_t size = n_[0];
    for (size_t i=1; i<d; ++i) {
      if (n_[i] == 0)
        n_[i] = n_[i-1];
      size *= n_[i];
    }
    if (size > 0) {
      data_ = new value_type[size];
      for (size_t i=0; i<size; ++i)
        data_[i] = value_type();
    }
  }
  
  template <int dim, typename U, typename... Args>
  void init(U&& i, Args&&... args) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside init(U&& i, Args&&... args) with index "<<i<<" at position "<<dim<<endl;
#endif
    n_[dim] = i;
    init<dim+1>(args...);
  }
  
  // parameter constructor
  template <typename... Args>
  explicit Array(const Args&... args) : base_type() {
    
#ifdef ARRAY_VERBOSE
    cout<<"Inside constructor template <typename... Args> explicit Array(const Args&... args)"<<endl;
#endif
    static_assert(sizeof...(Args) == d || sizeof...(Args) == 1 , "*** ERROR *** Wrong number of arguments for array");
    
    init<0>(args...);
  }
  
  
  Array(const Array& a) : base_type() {
    
#ifdef ARRAY_VERBOSE
    cout<<"Inside constructor Array(const Array& a)"<<endl;
#endif
    
    for (size_t i=0; i<d; ++i)
      n_[i] = a.n_[i];
    
    if(a.data_) {
      size_t s = size();
      data_ = new value_type[s];
      for(size_t i=0; i<s; ++i)
        data_[i] = a.data_[i];
    } else
      data_ = 0;
  }
  
  
  Array& operator=(std::initializer_list<value_type> l)
  {
    
    // set all elements to the value of the initializer list
    if (l.size() < 2) {
      value_type v = l.size() == 0 ? value_type() : *l.begin();
      for (iterator it = begin(); it != end(); ++it)
        *it = v;
    } else {
      
      iterator it = begin();
      typename std::initializer_list<value_type>::iterator lit = l.begin();
      cout<<"before"<<endl;
      while (it != end() && lit != l.end())
        *it++ = *lit++;
    }
    return *this;
  }
  
  
  
  Array& operator=(const Array& a) {
    
#ifdef ARRAY_VERBOSE
    cout<<"inside Array& operator=(const Array& a)"<<endl;
#endif
    
    // check for self-assignment
    if(this != &a) {
      
      for (int i=0; i<d; ++i) {
        n_[i] = a.n_[i];
        assert(n_[i] != 0);
      }
      
      // delete allocated memory
      delete[] data_;
      
      // check for null pointer
      if(a.data_) {
        size_t s = a.size();
        data_ = new value_type[s];
        for(size_t i=0; i<s; ++i)
          data_[i] = a.data_[i];
      } else
        data_ = 0;
    }
    return *this;
  }
  
  
  //    // move constructor
  //    Array(Array&& src) : base_type(std::forward<base_type>(src)) {
  //#ifdef ARRAY_VERBOSE
  //        cout<<"inside Array(Array&&)"<<endl;
  //#endif
  //    }
  
  
  //    // move constructor
  //    Array(Array&& src) {
  //
  //#ifdef ARRAY_VERBOSE
  //        cout<<"inside Array(Array&&)"<<endl;
  //#endif
  //        for (int i=0; i<d; ++i)
  //            n_[i] = src.n_[i];
  //        data_ = src.data_;
  //        src.data_ = NULL;
  //    }
  
  // constructor taking an expression with transpose operator
  Array(const Expr<BinExprOp<Array, EmptyType, ApTr> >& atr) {
    
#ifdef ARRAY_VERBOSE
    cout<<"inside constructor Array(const Expr<BinExprOp<Array, EmptyType, ApTr> >& atr)"<<endl;
#endif
    
    Array& a = *this;
    a = atr();
  }
  
  
  // constructor taking an arbitrary expression
  template <class A>
  Array(const Expr<A>& expr) {
    
#ifdef ARRAY_VERBOSE
    cout<<"inside template <class A> Array(const Expr<A>& expr)"<<endl;
#endif
    static_assert(SameClass<Array, typename A::result_type>::result, "*** ERROR *** Resulting expression is not of type array.");
    Array& a = *this;
    a = expr();
    
    //        execute(expr);
  }
  
  template <class A>
  typename enable_if<SameClass<Array, typename A::result_type>::result, void>::type
  execute(const Expr<A>& expr) {
    cout<<"INSIDE EXECUTE FOR MATRICES!"<<endl;
    
    Array& a = *this;
    a = expr();
  }
  
  template <class A>
  typename enable_if<!SameClass<Array, typename A::result_type>::result, void>::type
  execute(const Expr<A>& expr) {
    cout<<"INSIDE EXECUTE FOR NON MATRICES!"<<endl;
    
    static_assert(SameClass<Array, typename A::result_type>::result, "*** ERROR *** Resulting expression is not of type Array.");
  }
  
  //
  //    template <class A>
  //    enable_if<!IsArray<typename A::result_type> >::result, void>::type
  //    execute(const Expr<A>& expr) {
  //
  //        cout<<"INSIDE EXECUTE FOR OTHER STUFF!"<<endl;
  //
  //        Array& a = *this;
  //        a = expr();
  //    }
  
  
  //    // unary operator-(any)
  //    template <class A>
  //    typename enable_if<!is_arithmetic<A>::value, Expr<BinExprOp<ExprLiteral<int>, A, ApMul> > >::type
  //    operator-(const A& a) {
  //
  //#ifdef ARRAY_VERBOSE
  //        cout<<"1 Inside unary operator-(any), file "<<__FILE__<<", line "<<__LINE__<<endl;
  //        typedef BinExprOp<ExprLiteral<int>, A, ApMul> ExprT;
  //        cout<<"  expression type: "<<typeid(ExprT).name()<<endl;
  //#endif
  //        return (-1 *a);
  //    }
  
  ~Array() { delete data_; }
  
  // iterators
  iterator begin()
  { return iterator(data_); }
  
  //  const_iterator begin() const
  //  { return const_iterator(data_); }
  
  iterator end()
  { return iterator(data_ + size()); }
  
  //  const_iterator end() const
  //  { return const_iterator(data_ + size()); }
  
  reverse_iterator rbegin()
  { return reverse_iterator(end()); }
  
  //  const_reverse_iterator rbegin() const
  //  { return const_reverse_iterator(end()); }
  
  reverse_iterator rend()
  { return reverse_iterator(begin()); }
  
  //  const_reverse_iterator rend() const
  //  { return const_reverse_iterator(begin()); }
  
  size_t size() const {
    size_t n = 1;
    for (size_t i=0; i<d; ++i)
      n *= n_[i];
    return n;
  }
  
private:
  
  template <typename first_type, typename... Rest>
  class Check_integral {

    enum { tmp = std::is_integral<first_type>::value };

  public:

    typedef first_type pack_type;
    enum { value = tmp && Check_integral<Rest...>::value };
    static_assert (value ,"*** ERROR *** Non-integral type parameter found.");
  };
  
  template <typename last_type>
  struct Check_integral<last_type> {
    
    typedef last_type pack_type;
    enum { value = std::is_integral<last_type>::value };
  };

public:
  
  template <typename... Args>
  reference_type operator()(Args... params) {
    
    // check that the number of parameters corresponds to the size of the array
    static_assert(sizeof...(Args) == d , "*** ERROR *** Number of arguments does not match array dimension.");
    
    typedef typename Check_integral<Args...>::pack_type pack_type;
    
    // unpack parameters
    pack_type indices[] = { params... };
    
    // return reference
    return data_[index(indices)];
  }
  
  
  template <typename... Args>
  value_type operator()(Args... params) const {
    
    // check that the number of parameters corresponds to the size of the array
    static_assert(sizeof...(Args) == d , "*** ERROR *** Number of arguments does not match array dimension.");
    
    typedef typename Check_integral<Args...>::pack_type pack_type;

    // unpack parameters
    pack_type indices[] = { params... };
    
    // return reference
    return data_[index(indices)];
  }
  
  // support compound assignment operators for mathematical operations on arrays
  Array& operator*=(const value_type s) {
    cblas_dscal(size(), s, data_, 1);
    return *this;
  }
  
  Array& operator/=(const value_type s) {
    cblas_dscal(size(), value_type(1)/s, data_, 1);
    return *this;
  }
  
  Array& operator+=(const Array& b) {
    
    // check dimensions
    for (int i=0; i<d; ++i)
      assert(n_[i] = b.n_[i]);
    
    // call blas routine to add the arrays
    cblas_daxpy(size(), 1.0, b.data_, 1, data_, 1);
    // NOTE: the 1.0 is the factor by which v is scaled
    return *this;
  }
  
  Array& operator-=(const Array& b) {
    
    // check dimensions
    for (int i=0; i<d; ++i)
      assert(n_[i] = b.n_[i]);
    
    // call blas routine to add the arrays
    cblas_daxpy(size(), -1.0, b.data_, 1, data_, 1);
    return *this;
  }
  
  
  friend std::ostream& operator<<(std::ostream& os, const Array& a) {
    Print<Array::d_>::print(os, a.n_, a.data_);
    return os;
  }
  
  
  typedef typename Array_proxy_traits<d,Array>::reference_type proxy_reference;
  typedef typename Array_proxy_traits<d,Array>::value_type proxy_value;
  
  proxy_reference operator[](size_t i)
  { return Array_proxy_traits<d,Array>::reference(*this,i); }
  
  proxy_value operator[](size_t i) const
  { return Array_proxy_traits<d,Array>::value(*this,i); }
  
private:
  
  template <typename pack_type>
  pack_type index(pack_type indices[]) const {
    
    pack_type i = indices[0], s = 1;
    for (int j=1; j<d_; ++j) {
      assert(indices[j] > 0);
      assert(static_cast<size_t>(indices[j]) < n_[j]);
      // static cast to avoid compiler warning about comparison between signed and unsigned integers
      s *= n_[j-1];
      i += s * indices[j];
    }
    return i;
  }
  
  template <int, class>
  friend struct Array_proxy;
  
  friend class DApMul;
  friend class DApDivide;
};


// alias templates
template <class U>
using vector_type = array::Array<1,U>;

template <class U>
using matrix_type = array::Array<2,U>;

template <class U>
using tensor_type = array::Array<4,U>;


template <int d, class Array>
struct Array_proxy {
  
  typedef const Array_proxy<d-1, Array> value_type;
  typedef Array_proxy<d-1, Array> reference_type;
  
  //! Array constructor
  /*! The index is taken as the first component of operator[]
   * and no further update is needed
   */
  explicit Array_proxy (const Array& a, size_t i)
  : a_(a), i_(i), s_(a.n_[0]) {}
  
  template <int c>
  Array_proxy (const Array_proxy<c, Array>& a, size_t i)
  : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::d_ - c]) {}
  
  reference_type operator[](size_t i)
  { return reference_type(*this, i); }
  
  value_type operator[](size_t i) const
  { return value_type(*this, i); }
  
  const Array& a_;
  size_t i_;
  size_t s_;
};


template <class Array>
struct Array_proxy<1, Array> {
  
  typedef typename Array::reference_type reference_type;
  typedef typename Array::value_type value_type;
  
  explicit Array_proxy (const Array& a, size_t i)
  : a_(a), i_(i), s_(a.n_[0]) {}
  
  template <int c>
  Array_proxy (const Array_proxy<c, Array>& a, size_t i)
  : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::d_ - c]) {}
  
  reference_type operator[](size_t i)
  { return a_.data_[i_+i*s_]; }
  
  value_type operator[](size_t i) const
  { return a_.data_[i_+i*s_]; }
  
private:
  
  const Array& a_;
  size_t i_;
  size_t s_;
};


template <>
struct Print<1> {
  
  static std::ostream& print(std::ostream& os, const size_t size[], const double* data) {
    
    const size_t m = size[0];
    os<<"Array<1> ("<<m<<")"<<endl;
    for (size_t i=0; i<m; ++i)
      os<<" "<<data[i]<<endl;
    return os;
  }
};


template <>
struct Print<2> {
  
  static std::ostream& print(std::ostream& os, const size_t size[], const double* data) {
    
    const size_t m = size[0];
    const size_t n = size[1];
    
    os<<"Array<2> ("<<m<<"x"<<n<<")"<<endl;
    for (size_t i=0; i<m; ++i) {
      for (size_t j=0; j<n; ++j)
        os<<" "<<data[i + j*m];
      cout<<endl;
    }
    return os;
  }
};


template <int d>
struct Print {
  
  static std::ostream& print(std::ostream& os, const size_t size[], const double* data) {
    
    size_t s = 1;
    for (int i=0; i<d-1; ++i)
      s *= size[i];
    
    for (size_t i=0; i<size[d-1]; ++i) {
      os<<"Dim "<<d<<": "<<i<<", ";
      Print<d-1>::print(os, size, data + i*s);
    }
    return os;
  }
};

__END_ARRAY_NAMESPACE__

#endif /* ARRAY_HPP */
