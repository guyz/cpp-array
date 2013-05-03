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
#include <cmath>
#include <algorithm>


#include "return_type.hpp"


__BEGIN_ARRAY_NAMESPACE__

using std::cout;
using std::endl;


template <int d>
struct Print;

template <int d, class Array>
struct Array_proxy;

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



template <typename first_type, typename... Rest>
class Check_first {
  
public:
  
  typedef first_type pack_type;
  enum { value = std::is_integral<first_type>::value};
};


template <typename last_type>
struct Check_first<last_type> {
  
  typedef last_type pack_type;
  enum { value = std::is_integral<last_type>::value };
};


template <int d, class Array>
struct Array_proxy_traits {
  
  typedef Array_proxy<d-1, Array> reference;
  typedef const Array_proxy<d-1, Array> value_type;
  
  static reference get_reference(Array& a, size_t i)
  { return reference(a,i); }
  
  static value_type value(Array& a, size_t i)
  { return value_type(a,i); }
};


template <class Array>
struct Array_proxy_traits<1,Array> {
  typedef typename Array::value_type primitive_type;
  typedef primitive_type& reference;
  typedef primitive_type const & value_type;
  
  static reference get_reference(Array& a, size_t i)
  { return a.data_[i]; }
  
  static value_type value(Array& a, size_t i)
  { return a.data_[i]; }
};



template <class A, class B>
struct SameClass {
  enum { result = false };
};

template <class A>
struct SameClass<A,A> {
  enum { result = true };
};


template <typename T, typename P, int d = -1>
class Array_iterator;

template <typename T, typename P, int d>
class Array_iterator : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, P> {
  
public:
  
  typedef P pointer;
  
  constexpr static int dim()
  { return d; }
  
  Array_iterator(T* x, size_t str) :p_(x), str_(str) {}
  Array_iterator(const Array_iterator& mit) : p_(mit.p_), str_(mit.str_) {}
  
  // Allow iterator to const_iterator conversion
  template<typename iterator>
  Array_iterator(const Array_iterator<T,
                 typename std::enable_if<(SameClass<P, typename iterator::pointer>::result), P>::type>& i)
  : p_(i.p_), str_(i.str_) {}
  
  Array_iterator& operator++()
  { p_ += str_; return *this; }
  
  Array_iterator operator++(int)
  { Array_iterator it(p_);
    p_ += str_;
    return it; }
  
  Array_iterator& operator--()
  { p_ -= str_; return *this; }
  
  Array_iterator operator--(int)
  { Array_iterator it(p_);
    p_ -= str_;
    return it; }
  
  bool operator==(const Array_iterator& rhs)
  {return p_ == rhs.p_;}
  
  bool operator!=(const Array_iterator& rhs)
  {return p_ != rhs.p_;}
  
  T& operator*()
  {return *p_;}
  
private:
  
  pointer p_;   //!< Pointer
  size_t str_;  //!< Stride
};



template <typename T, typename P>
class Array_iterator<T, P, -1> : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, P> {
  
public:
  
  typedef P pointer;
  
  Array_iterator(T* x) :p_(x) {}
  Array_iterator(const Array_iterator& mit) : p_(mit.p_) {}
  
  // Allow iterator to const_iterator conversion
  template<typename iterator>
  Array_iterator(const Array_iterator<T,
                 typename std::enable_if<(SameClass<P, typename iterator::pointer>::result), P>::type>& i)
  : p_(i.p_) {}
  
  Array_iterator& operator++()
  { ++p_; return *this; }
  
  Array_iterator operator++(int)
  { return Array_iterator(p_++); }
  
  Array_iterator& operator--()
  { --p_; return *this; }
  
  Array_iterator operator--(int)
  { return Array_iterator(p_--); }
  
  bool operator==(const Array_iterator& rhs)
  {return p_ == rhs.p_;}
  
  bool operator!=(const Array_iterator& rhs)
  {return p_ != rhs.p_;}
  
  T& operator*()
  {return *p_;}
  
private:
  
  pointer p_;
};




template <int k, typename T, class array_type>
class Array_traits {
  
  
};


template <typename T, class array_type>
class Array_traits<1,T, array_type> {
  
protected:
  
  typedef T value_type;
  
  template <class functor>
  void fill(functor fn) {
    array_type &a = static_cast<array_type&>(*this);
    for (size_t i=0; i<a.n_[0]; ++i)
      a.data_[i] = fn(i);
  }
  
public:
  
  value_type norm() const {
    assert(this->n_[0] > 0);
    return norm(Type2Type<value_type>());
  }
  
private:
  
  template <class U>
  inline U norm(Type2Type<U>) const {
    U norm = U();
    array_type &a = static_cast<array_type&>(*this);
    for (size_t i=0; i<a.n_[0]; ++i)
      norm += std::pow(a.data_[i],2);
    norm = std::sqrt(norm);
    return norm;
  }
  
  // norm function
  inline double norm(Type2Type<double>) const {
    // call to blas routine
    return cblas_dnrm2(this->n_[0], this->data_, 1);
  }
};


template <typename T, class array_type>
class Array_traits<2,T, array_type> {
  
protected:
  
  typedef T value_type;
  
  typedef std::initializer_list<T> list_type;
  
  template <class functor>
  void fill(functor fn) {
    array_type &a = static_cast<array_type&>(*this);
    for (size_t i=0; i<rows(); ++i)
      for (size_t j=0; j<columns(); ++j)
        a.data_[i + j*rows()] = fn(i,j);
  }
  
public:
  
  size_t rows() const
  { return static_cast<array_type const *>(this)->n_[0]; }
  
  size_t columns() const
  { return static_cast<array_type const *>(this)->n_[1]; }
};


template <typename T, class array_type>
class Array_traits<4, T, array_type> {
  
protected:
  
  typedef T value_type;
  
  template <class functor>
  void fill(functor fn) {
    
    array_type &a = static_cast<array_type&>(*this);
    size_t m = a.n_[0];
    size_t n = a.n_[1];
    size_t o = a.n_[2];
    
    for (size_t i=0; i<m; ++i)
      for (size_t j=0; j<n; ++j)
        for (size_t k=0; k<o; ++k)
          for (size_t l=0; l<a.n_[3]; ++l)
            a.data_[i + m*j + m*n*k + m*n*o*l] = fn(i,j,k,l);
  }

};





template <int k, typename T>
class Array : public Array_traits<k,T, Array<k,T> > {
  
  typedef Array_traits<k,T, Array> traits_type;
  
public:
  
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  
  typedef Array_iterator<value_type, pointer> iterator;
  typedef Array_iterator<const value_type, const_pointer> const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  
  
private:
  
  size_t n_[k] = {0};
  pointer data_;
  bool wrapped_;
  
public:
  
  static int rank() { return k; }
  
  // default constructor
  Array() : data_(nullptr), wrapped_() {
    
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array()"<<endl;
#endif
  }
  
  // copy constructor
  Array(const Array& a);
  
  // move constructor
  Array(Array&& src);
  
  // destructor
  ~Array() {
#ifdef ARRAY_VERBOSE
    cout<<"Inside ~Array()"<<endl;
#endif
    if (!wrapped_) delete[] data_; }
  
  // assignment operator
  Array& operator=(const Array& src);
  
  // move assignment operator
  Array& operator=(Array&& src);
  
private:
  
  size_t init_dim() {
    size_t s = n_[0];
    for (size_t i=1; i<k; ++i) {
      if (n_[i] == 0)
        n_[i] = n_[i-1];
      s *= n_[i];
    }
    return s;
  }
  
  // init takes an integer parameter
  template <int d, typename U, typename... Args>
  typename std::enable_if<std::is_integral<U>::value and !std::is_pointer<U>::value and d < k, void>::type
  init(U i, Args&&... args) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array::init(int i, Args&&... args) with index n["<<d<<"] = "<<i<<endl;
#endif
    assert(i != 0); // Array dimension cannot be zero
    n_[d] = i;
    init<d+1>(args...);
  }
  
  // init takes a value to initialize all elements
  template <int d>
  void init(value_type v = value_type()) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array::init(value_type), with value: "<<v<<endl;
#endif
    size_t s = init_dim();
    data_ = new value_type[s];
    std::fill_n(data_, s, v);
  }
  
  // init with a pointer to already existing data
  template <int d, typename P, typename... Args>
  typename std::enable_if<std::is_pointer<P>::value, void>::type
  init(P p, Args&&... args) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array::init(U* p, Args&&... args) for pointer"<<endl;
#endif
    init_dim();
    wrapped_ = true;
    data_ = p;
  }
  
  // init takes a functor, lambda expression, etc.
  template <int d, class functor>
  typename std::enable_if<!std::is_integral<functor>::value and !std::is_pointer<functor>::value, void>::type
  init(functor fn) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array::init(std::function)"<<endl;
#endif
    size_t s = init_dim();
    data_ = new value_type[s];
    this->fill(fn);
  }

  
public:
  
  // parameter constructor
  template <typename... Args>
  Array(const Args&... args) : data_(nullptr), wrapped_() {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array(const Args&... args)"<<endl;
#endif
    static_assert(sizeof...(Args) <= k+1, "*** ERROR *** Wrong number of arguments for array");
    init<0>(args...);
  }
  
  
  
  template <int d, typename U>
  struct Initializer_list {

    typedef std::initializer_list<typename Initializer_list<d-1,U>::list_type > list_type;
    
    static void process(list_type l, Array& a, size_t s, size_t idx) {

      a.n_[k-d] = l.size(); // set dimension

      size_t j = 0;
      for (const auto& r : l)
        Initializer_list<d-1, U>::process(r, a, s*l.size(), idx + s*j++);
    }
  };
  
  // partial template specialization to finish recursion
  template <typename U>
  struct Initializer_list<1,U> {

    typedef std::initializer_list<U> list_type;

    static void process(list_type l, Array& a, size_t s, size_t idx) {

      a.n_[k-1] = l.size(); // set dimension
      if (!a.data_)
        a.data_ = new value_type[s*l.size()];
      
      size_t j = 0;
      for (const auto& r : l)
        a.data_[idx + s*j++] = r;
    }
  };
  
  typedef typename Initializer_list<k,T>::list_type initializer_type;

  // initializer list constructor
  Array(initializer_type l) : wrapped_(), data_(nullptr) {
#ifdef ARRAY_VERBOSE
    cout<<"Inside Array<1,T>(initializer_list)"<<endl;
#endif
    Initializer_list<k, T>::process(l, *this, 1, 0);
  }


  // constructor taking an arbitrary expression
  template <class A>
  Array(const Expr<A>& expr) : data_(nullptr), wrapped_() {
    
#ifdef ARRAY_VERBOSE
    cout<<"Inside template <class A> Array(const Expr<A>& expr)"<<endl;
#endif
    static_assert(SameClass<Array, typename A::result_type>::result, "*** ERROR *** Resulting expression is not of type array.");
    Array& a = *this;
    a = expr();
  }
  
  // functions
  
  size_t size() const {
    size_t n = 1;
    for (size_t i=0; i<k; ++i)
      n *= n_[i];
    return n;
  }
  
  size_t size(size_t i) const
  { return n_[i]; }
  
  
  // indexed access operators
  
private:
  
  template <typename pack_type>
  pack_type index(pack_type indices[]) const {
    
    pack_type i = indices[0], s = 1;
    for (int j=1; j<k; ++j) {
      assert(indices[j] >= 0);
      assert(static_cast<size_t>(indices[j]) < n_[j]);
      // static cast to avoid compiler warning about comparison between signed and unsigned integers
      s *= n_[j-1];
      i += s * indices[j];
    }
    return i;
  }
  
  template <typename first_type, typename... Rest>
  struct Check_integral {
    typedef first_type pack_type;
    enum { tmp = std::is_integral<first_type>::value };
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
  reference operator()(Args... params) {
    
    // check that the number of parameters corresponds to the size of the array
    static_assert(sizeof...(Args) == k , "*** ERROR *** Number of parameters does not match array rank.");
    
    typedef typename Check_integral<Args...>::pack_type pack_type;
    
    // unpack parameters
    pack_type indices[] = { params... };
    
    // return reference
    return data_[index(indices)];
  }
  
  
  template <typename... Args>
  value_type operator()(Args... params) const {
    
    // check that the number of parameters corresponds to the size of the array
    static_assert(sizeof...(Args) == k , "*** ERROR *** Number of parameters does not match array rank.");
    
    typedef typename Check_integral<Args...>::pack_type pack_type;
    
    // unpack parameters
    pack_type indices[] = { params... };
    
    // return reference
    return data_[index(indices)];
  }
  
  typedef typename Array_proxy_traits<k,Array>::reference proxy_reference;
  typedef typename Array_proxy_traits<k,Array>::value_type proxy_value;
  
  proxy_reference operator[](size_t i)
  { return Array_proxy_traits<k,Array>::get_reference(*this,i); }
  
  proxy_value operator[](size_t i) const
  { return Array_proxy_traits<k,Array>::value(*this,i); }
  
  
  
  // compound assignment operators
  
  Array& operator*=(value_type s) {
    cblas_dscal(size(), s, data_, 1);
    return *this;
  }
  
  Array& operator/=(value_type s) {
    cblas_dscal(size(), value_type(1)/s, data_, 1);
    return *this;
  }
  
  Array& operator+=(const Array& b) {
    
    // check dimensions
    for (int i=0; i<k; ++i)
      assert(n_[i] == b.n_[i]);
    
    // call blas routine to add the arrays
    cblas_daxpy(size(), 1.0, b.data_, 1, data_, 1);
    // NOTE: the 1.0 is the factor by which v is scaled
    return *this;
  }
  
  Array& operator-=(const Array& b) {
    
    // check dimensions
    for (int i=0; i<k; ++i)
      assert(n_[i] == b.n_[i]);
    
    // call blas routine to add the arrays
    cblas_daxpy(size(), -1.0, b.data_, 1, data_, 1);
    return *this;
  }

  
  size_t stride(size_t dim) const {
    
    size_t s = 1;
    for (int j=0; j<dim; ++j)
      s *= n_[j];
    return s;
  }
  
  // iterators
  iterator begin()
  { return iterator(data_); }
  
  const_iterator begin() const
  { return const_iterator(data_); }
  
  iterator end()
  { return iterator(data_ + size()); }
  
  const_iterator end() const
  { return const_iterator(data_ + size()); }
  
  reverse_iterator rbegin()
  { return reverse_iterator(end()); }
  
  const_reverse_iterator rbegin() const
  { return const_reverse_iterator(end()); }
  
  reverse_iterator rend()
  { return reverse_iterator(begin()); }
  
  const_reverse_iterator rend() const
  { return const_reverse_iterator(begin()); }
  
  
  template <int d>
  using diterator = Array_iterator<value_type, pointer, d>;
  
  // dimensional iterators
  template <int d>
  diterator<d> dbegin()
  { return diterator<d>(data_, stride(d)); }
  
  template <int d>
  diterator<d> dend()
  { size_t s = stride(d); return diterator<d>(data_ + stride(d+1), s); }
  
  template <int d, typename iterator>
  diterator<d> dbegin(iterator it)
  { return diterator<d>(&*it, stride(d)); }
  
  template <int d, typename iterator>
  diterator<d> dend(iterator it)
  { size_t s = stride(d); return diterator<d>(&*it + stride(d+1), s); }
  
  
  template <int d>
  diterator<d> dbegin() const
  { return diterator<d>(data_, stride(d)); }
  
  template <int d>
  diterator<d> dend() const
  { size_t s = stride(d); return diterator<d>(data_ + stride(d+1), s); }
  
  template <int d, typename iterator>
  diterator<d> dbegin(iterator it) const
  { return diterator<d>(&*it, stride(d)); }
  
  template <int d, typename iterator>
  diterator<d> dend(iterator it) const
  { size_t s = stride(d); return diterator<d>(&*it + stride(d+1), s); }
  
  
  
  // friend classes and functions
  
  friend class Array_traits<k,T, Array>;

  template <int dim, class array_type>
  friend class Array_proxy;

  friend Array_proxy_traits<k,Array>;
  friend class ApAdd;
  friend class ApSub;
  friend class ApMul;
  
  friend std::ostream& operator<<(std::ostream& os, const Array& a) {
    if (a.size() == 0) {
      os<<"Empty array"<<endl;
      return os;
    }
    if (a.wrapped_)
      os<<"Wrapped ";
//    Print<k>::print(os, a, a.dbegin<0>(), a.dend<0>());
    Print<k>::print(os, a.n_, a.data_);
    return os;
  }
  
};


// copy constructor
template <int k, typename T>
Array<k,T>::Array(const Array<k,T>& a) : data_(nullptr), wrapped_() {
  
  
#ifdef ARRAY_VERBOSE
  cout<<"Inside Array(const Array&)"<<endl;
#endif
  
  std::copy_n(a.n_, k, n_);
  wrapped_ = a.wrapped_;
  
  if (!wrapped_) {
    size_t s = size();
    assert((a.data_ && s > 0) || (!a.data_ && s == 0));
    
    if (a.data_) {
      data_ = new value_type[s];
      std::copy_n(a.data_, s, data_);
    } else
      data_ = nullptr;
  } else
    data_ = a.data_;
}

// move constructor
template <int k, typename T>
Array<k,T>::Array(Array<k,T>&& src) : data_(nullptr), wrapped_() {
  
#ifdef ARRAY_VERBOSE
  cout<<"Inside Array(Array&&)"<<endl;
#endif
  
  std::copy_n(src.n_, k, n_);
  data_ = src.data_;
  wrapped_ = src.wrapped_;
  
  std::fill_n(src.n_, k, 0);
  src.data_ = nullptr;
  src.wrapped_ = false;
}


// assignment operator
template <int k, typename T>
Array<k,T>& Array<k,T>::operator=(const Array<k,T>& src) {
  
#ifdef ARRAY_VERBOSE
  cout<<"Inside operator=(const Array& src)"<<endl;
#endif
  if (this != &src) {
    
    if (!wrapped_)
      delete[] data_;
    
    std::copy_n(src.n_, k, n_);
    wrapped_ = src.wrapped_;
    
    if (!wrapped_) {
      
      size_t s = size();
      assert((src.data_ && s > 0) || (!src.data_ && s == 0));
      
      if (src.data_) {
        data_ = new value_type[s];
        std::copy_n(src.data_, s, data_);
      } else
        data_ = nullptr;
    } else
      data_ = src.data_;
  }
  return *this;
}

// move assignment operator
template <int k, typename T>
Array<k,T>& Array<k,T>::operator=(Array<k,T>&& src) {
  
#ifdef ARRAY_VERBOSE
  cout<<"Inside operator=(Array&& src)"<<endl;
#endif
  
  if (this != &src) {
    
    if (!wrapped_) delete data_;
    
    std::copy_n(src.n_, k, n_);
    wrapped_ = src.wrapped_;
    data_ = src.data_;
    
    // set src to default
    src.data_ = nullptr;
    src.wrapped_ = false;
    std::fill_n(src.n_, k, 0);
  }
  return *this;
}




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
  typedef Array_proxy<d-1, Array> reference;
  
  //! Array constructor
  /*! The index is taken as the first component of operator[]
   * and no further update is needed
   */
  explicit Array_proxy (const Array& a, size_t i)
  : a_(a), i_(i), s_(a.n_[0]) {}
  
  template <int c>
  Array_proxy (const Array_proxy<c, Array>& a, size_t i)
  : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::rank() - c]) {}
  
  reference operator[](size_t i)
  { return reference(*this, i); }
  
  value_type operator[](size_t i) const
  { return value_type(*this, i); }
  
  const Array& a_;
  size_t i_, s_;
};


template <class Array>
struct Array_proxy<1, Array> {
  
  typedef typename Array::reference reference;
  typedef typename Array::value_type value_type;
  
  explicit Array_proxy (const Array& a, size_t i)
  : a_(a), i_(i), s_(a.n_[0]) {}
  
  template <int c>
  Array_proxy (const Array_proxy<c, Array>& a, size_t i)
  : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::rank() - c]) {}
  
  reference operator[](size_t i)
  { return a_.data_[i_+i*s_]; }
  
  value_type operator[](size_t i) const
  { return a_.data_[i_+i*s_]; }
  
private:
  
  const Array& a_;
  size_t i_, s_;
};


template <>
struct Print<1> {

//  template <typename value_type>
//  static std::ostream& print(std::ostream& os, const vector_type<value_type>& a) {
//    
//    os<<"Array<1> ("<<a.size()<<")"<<endl;
//    for (typename vector_type<value_type>::iterator it = a.begin();
//         it != a.end(); ++it)
//      os<<" "<<*it<<endl;
//    return os;
//  }

  
  template <typename value_type>
  static std::ostream& print(std::ostream& os, const size_t size[], const value_type* data) {
    
    const size_t m = size[0];
    os<<"Array<1> ("<<m<<")"<<endl;
    for (size_t i=0; i<m; ++i)
      os<<' '<<data[i]<<'\n';
    return os;
  }
};


template <>
struct Print<2> {
  
  template<class array, typename iterator>
  static std::ostream& print(std::ostream& os, const array& A, iterator a, iterator b) {
    
    os<<"Array<2> ("<<A.size(0)<<"x"<<A.size(1)<<")"<<endl;

    constexpr int d = iterator::dim()+1;
    
    for (; a != b; ++a) {
      for (auto it = A.template dbegin<d>(a); it != A.template dend<d>(a); ++it)
        cout<<' '<<*it;
      cout<<'\n';
    }
    return os;
  }

  
  
  template <typename value_type>
  static std::ostream& print(std::ostream& os, const size_t size[], const value_type* data) {
    
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
  
  template<class array, typename iterator>
  static std::ostream& print(std::ostream& os, const array& A, iterator a, iterator b) {
    
    int i = 0;
    for (; a != b; ++a) {
//      for (auto it = A.template dbegin<d>(a); it != A.template dend<d>(a); ++it) {
        os<<"Dim "<<d<<": "<<i++<<", ";
        Print<d-1>::print(os, A, A.template dbegin<d-1>(a), A.template dend<d-1>(a));
      }
//    }

    return os;
  }


  template <typename value_type>
  static std::ostream& print(std::ostream& os, const size_t size[], const value_type* data) {
    
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
