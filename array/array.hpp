/*
 * Copyright (©) 2014 Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
 * All Rights Reserved
 *
 * cpp-array is free  software: you can redistribute it and/or  modify it under
 * the terms  of the  GNU Lesser  General Public  License as  published by  the
 * Free Software Foundation, either version 3 of the License, or (at your
 *option)
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

/*! \file array.hpp
 *
 * \brief This file presents the implementaiton of the arbitrary-rank
 * tensor from a single Array class template.
 */

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <iostream>
#include <cassert>
#include <iomanip>
#include <initializer_list>
#include <cmath>
#include <algorithm>

#include "return_type.hpp"
#include "blas.hpp"

__BEGIN_ARRAY_NAMESPACE__

using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////
// forward declarations

template <int d> struct Print;

template <int d, class Array> struct Array_proxy;

////////////////////////////////////////////////////////////////////////////////
// helper classes

//! int 2 type construct
template <int d> struct Int2Type {
  enum {
    result = d
  };
};

//! type 2 type construct
template <class T> class Type2Type {
  typedef T OriginalType;
};

//! Class construct for different classes
template <class A, class B> struct SameClass {
  enum {
    result = false
  };
};

//! Class construct for the same class
template <class A> struct SameClass<A, A> {
  enum {
    result = true
  };
};

//! Array proxy traits class template
/*! This class is used by the array class template to define
 * the proxy classes used for operator[].
 */
template <int d, class Array> struct Array_proxy_traits {

  typedef Array_proxy<d - 1, Array> reference;
  typedef const Array_proxy<d - 1, Array> value_type;

  static reference get_reference(Array &a, size_t i) { return reference(a, i); }

  static value_type value(const Array &a, size_t i) { return value_type(a, i); }
};

//! Array proxy traits class template
/*! This class is used by the array class template to define
 * the proxy classes used for operator[].
 * This class is used to finish the compile-time recursion.
 */
template <class Array> struct Array_proxy_traits<1, Array> {
  typedef typename Array::value_type primitive_type;
  typedef primitive_type &reference;
  typedef primitive_type const &value_type;

  static reference get_reference(Array &a, size_t i) { return a.data_[i]; }

  static value_type value(const Array &a, size_t i) { return a.data_[i]; }
};

////////////////////////////////////////////////////////////////////////////////
// iterator classes

template <typename T, typename P, int d = -1> class Array_iterator;

//! Array iterator class template
/*! This class is used to iterate over any dimension of the array.
 * The class contains as data members a pointer to memory, and the
 * stride that will be used to advance the iterator.
 */
template <typename T, typename P, int d>
class Array_iterator : public std::iterator<std::random_access_iterator_tag, T,
                                            std::ptrdiff_t, P> {

public:
  typedef P pointer;

  //! Iterator dimension
  static int dim() { return d; }

  //! Pointer and stride parameter constructor
  Array_iterator(T *x, size_t str) : p_(x), str_(str) {}

  //! Copy constructor
  Array_iterator(const Array_iterator &mit) : p_(mit.p_), str_(mit.str_) {}

  // Allow iterator to const_iterator conversion
  template <typename iterator>
  Array_iterator(const Array_iterator<
      T, typename std::enable_if<
             (SameClass<P, typename iterator::pointer>::result), P>::type> &i)
      : p_(i.p_), str_(i.str_) {}

  //! Pre-increment iterator
  Array_iterator &operator++() {
    p_ += str_;
    return *this;
  }

  //! Post-increment iterator
  Array_iterator operator++(int) {
    Array_iterator it(p_);
    p_ += str_;
    return it;
  }

  //! Pre-decrement iterator
  Array_iterator &operator--() {
    p_ -= str_;
    return *this;
  }

  //! Post-decrement iterator
  Array_iterator operator--(int) {
    Array_iterator it(p_);
    p_ -= str_;
    return it;
  }

  //! Equal-to operator
  bool operator==(const Array_iterator &rhs) { return p_ == rhs.p_; }

  //! Not-equal-to operator
  bool operator!=(const Array_iterator &rhs) { return p_ != rhs.p_; }

  //! Dereference operator
  T &operator*() { return *p_; }

private:
  pointer p_;  //!< Pointer to memory
  size_t str_; //!< Stride
};

//! Array iterator class template
/*! This partial template specialization is used to iterate over all
 * the elements of the array.
 */
template <typename T, typename P>
class Array_iterator<T, P,
                     -1> : public std::iterator<std::random_access_iterator_tag,
                                                T, std::ptrdiff_t, P> {

public:
  typedef P pointer;

  //! Pointer parameter constructor
  Array_iterator(T *x) : p_(x) {}

  //! Copy constructor
  Array_iterator(const Array_iterator &mit) : p_(mit.p_) {}

  // Allow iterator to const_iterator conversion
  template <typename iterator>
  Array_iterator(const Array_iterator<
      T, typename std::enable_if<
             (SameClass<P, typename iterator::pointer>::result), P>::type> &i)
      : p_(i.p_) {}

  //! Pre-increment iterator
  Array_iterator &operator++() {
    ++p_;
    return *this;
  }

  //! Post-increment iterator
  Array_iterator operator++(int) { return Array_iterator(p_++); }

  //! Pre-decrement iterator
  Array_iterator &operator--() {
    --p_;
    return *this;
  }

  //! Post-decrement iterator
  Array_iterator operator--(int) { return Array_iterator(p_--); }

  //! Equal-to operator
  bool operator==(const Array_iterator &rhs) { return p_ == rhs.p_; }

  //! Not-equal-to operator
  bool operator!=(const Array_iterator &rhs) { return p_ != rhs.p_; }

  //! Dereference operator
  T &operator*() { return *p_; }

private:
  pointer p_; //!< Pointer to memory
};

////////////////////////////////////////////////////////////////////////////////
// array traits clases

//! Array traits class
template <int k, typename T, class array_type> class Array_traits {};

//! Array traits partial template specialization for vectors
template <typename T, class array_type> class Array_traits<1, T, array_type> {

protected:
  typedef T value_type;

  //! Helper function used to fill a vector from a functor, lambda expression,
  //etc.
  template <class functor> void fill(functor fn) {
    array_type &a = static_cast<array_type &>(*this);
    for (size_t i = 0; i < a.n_[0]; ++i)
      a.data_[i] = fn(i);
  }

public:
  //! Vector L2 norm
  /*! This function calls a helper function depending on the type
   * stored in the vector.
   */
  value_type norm() const { return norm(Type2Type<value_type>()); }

private:
  //! Helper function used by norm
  template <class U> inline U norm(Type2Type<U>) const {
    U norm = U();
    array_type &a = static_cast<array_type &>(*this);
    assert(a.n_[0] > 0);
    for (size_t i = 0; i < a.n_[0]; ++i)
      norm += std::pow(a.data_[i], 2);
    norm = std::sqrt(norm);
    return norm;
  }

  //! Helper function used by norm when storing double type
  inline double norm(Type2Type<double>) const {
    // call to blas routine
    const array_type &a = static_cast<const array_type &>(*this);
    return cblas_nrm2(a.n_[0], a.data_, 1);
  }
};

//! Array traits partial template specialization for matrices
template <typename T, class array_type> class Array_traits<2, T, array_type> {

protected:
  typedef T value_type;

  typedef std::initializer_list<T> list_type;

  //! Helper function used to fill a vector from a functor, lambda expression,
  //etc.
  template <class functor> void fill(functor fn) {
    array_type &a = static_cast<array_type &>(*this);
    for (size_t i = 0; i < rows(); ++i)
      for (size_t j = 0; j < columns(); ++j)
        a.data_[i + j * rows()] = fn(i, j);
  }

public:
  //! Matrix rows
  size_t rows() const { return static_cast<array_type const *>(this)->n_[0]; }

  //! Matrix columns
  size_t columns() const {
    return static_cast<array_type const *>(this)->n_[1];
  }
};

//! Array traits partial template specialization for 4th order tensors
template <typename T, class array_type> class Array_traits<4, T, array_type> {

protected:
  typedef T value_type;

  //! Helper function used to fill a vector from a functor, lambda expression,
  //etc.
  template <class functor> void fill(functor fn) {

    array_type &a = static_cast<array_type &>(*this);
    size_t m = a.n_[0];
    size_t n = a.n_[1];
    size_t o = a.n_[2];

    for (size_t i = 0; i < m; ++i)
      for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < o; ++k)
          for (size_t l = 0; l < a.n_[3]; ++l)
            a.data_[i + m * j + m * n * k + m * n * o * l] = fn(i, j, k, l);
  }
};

////////////////////////////////////////////////////////////////////////////////
// array class template

//! Array class template
/*! This class template is used to define tensors of any rank.
 * \tparam k - Tensor rank
 * \tparam T - Type stored in the tensor
 *
 * The class template inherits from Array_traits passing itself as
 * a template parameter to use the Curiously Recurring Template
 * Pattern (CRTP).
 */
template <int k, typename T>
class Array : public Array_traits<k, T, Array<k, T> > {

  typedef Array_traits<k, T, Array> traits_type;

public:
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;

  typedef Array_iterator<value_type, pointer> iterator;
  typedef Array_iterator<const value_type, const_pointer> const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

private:
  size_t n_[k] = { 0 }; //!< Tensor dimensions
  pointer data_;        //!< Pointer to memory
  bool wrapped_;        //!< Owned memory flag

public:
  //! Rank of the tensor
  static int rank() { return k; }

  //! Pointer to memory for raw access
  pointer data() { return data_; }

  //! Default constructor
  Array() : data_(nullptr), wrapped_() {}

  //! Copy constructor
  Array(const Array &a);

  //! Move constructor
  Array(Array &&src);

  //! Destructor
  ~Array() {
    if (!wrapped_)
      delete[] data_;
  }

  //! Assignment operator
  Array &operator=(const Array &src);

  //! Move assignment operator
  Array &operator=(Array &&src);

private:
  //! Helper function used by constructors
  size_t init_dim() {
    size_t s = n_[0];
    for (size_t i = 1; i < k; ++i) {
      if (n_[i] == 0)
        n_[i] = n_[i - 1];
      s *= n_[i];
    }
    return s;
  }

  //! init helper function that takes an integer parameter
  template <int d, typename U, typename... Args>
      typename std::enable_if <
      std::is_integral<U>::value and !std::is_pointer<U>::value and
      d<k, void>::type init(U i, Args && ... args) {

    assert(i != 0); // Array dimension cannot be zero
    n_[d] = i;
    init<d + 1>(args...);
  }

  //! init helper function that takes a value to initialize all elements
  template <int d> void init(value_type v = value_type()) {

    size_t s = init_dim();
    data_ = new value_type[s];
    std::fill_n(data_, s, v);
  }

  //! init helper function that takes a pointer to already existing data
  template <int d, typename P, typename... Args>
  typename std::enable_if<std::is_pointer<P>::value, void>::type
  init(P p, Args &&... args) {

    init_dim();
    wrapped_ = true;
    data_ = p;
  }

  //! init helper function that takes a functor, lambda expression, etc.
  template <int d, class functor>
  typename std::enable_if<
      !std::is_integral<functor>::value and !std::is_pointer<functor>::value,
      void>::type
  init(functor fn) {

    size_t s = init_dim();
    data_ = new value_type[s];
    this->fill(fn);
  }

public:
  //! Parameter constructor, uses the init helper function
  template <typename... Args>
  Array(const Args &... args)
      : data_(nullptr), wrapped_() {

    static_assert(sizeof...(Args) <= k + 1,
                  "*** ERROR *** Wrong number of arguments for array");
    init<0>(args...);
  }

  //! Helper structure used to process initializer lists
  template <int d, typename U> struct Initializer_list {

    typedef std::initializer_list<
        typename Initializer_list<d - 1, U>::list_type> list_type;

    static void process(list_type l, Array &a, size_t s, size_t idx) {

      a.n_[k - d] = l.size(); // set dimension
      size_t j = 0;
      for (const auto &r : l)
        Initializer_list<d - 1, U>::process(r, a, s * l.size(), idx + s * j++);
    }
  };

  //! Helper structure used to process initializer lists, partial template
  //specialization to finish recursion
  template <typename U> struct Initializer_list<1, U> {

    typedef std::initializer_list<U> list_type;

    static void process(list_type l, Array &a, size_t s, size_t idx) {

      a.n_[k - 1] = l.size(); // set dimension
      if (!a.data_)
        a.data_ = new value_type[s * l.size()];

      size_t j = 0;
      for (const auto &r : l)
        a.data_[idx + s * j++] = r;
    }
  };

  typedef typename Initializer_list<k, T>::list_type initializer_type;

  //! Initializer list constructor
  Array(initializer_type l) : data_(nullptr), wrapped_() {

    Initializer_list<k, T>::process(l, *this, 1, 0);
  }

  //! constructor taking an arbitrary expression
  template <class A> Array(const Expr<A> &expr) : data_(nullptr), wrapped_() {

    static_assert(SameClass<Array, typename A::result_type>::result,
                  "*** ERROR *** Resulting expression is not of type array.");
    Array &a = *this;
    a = expr();
  }

  //! Size of the tensor
  size_t size() const {
    size_t n = 1;
    for (size_t i = 0; i < k; ++i)
      n *= n_[i];
    return n;
  }

  //! Size along the ith direction
  size_t size(size_t i) const { return n_[i]; }

private:
  ////////////////////////////////////////////////////////////////////////////////
  // indexed access operators

  //! Helper function used to compute the index on the one-dimensional array
  //that stores the tensor elements
  template <typename pack_type> pack_type index(pack_type indices[]) const {

    pack_type i = indices[0], s = 1;
    for (int j = 1; j < k; ++j) {
      assert(indices[j] >= 0);
      assert(static_cast<size_t>(indices[j]) < n_[j]);
      // static cast to avoid compiler warning about comparison between signed
      // and unsigned integers
      s *= n_[j - 1];
      i += s * indices[j];
    }
    return i;
  }

  //! Helper structure used by operator()
  template <typename first_type, typename... Rest> struct Check_integral {
    typedef first_type pack_type;
    enum {
      tmp = std::is_integral<first_type>::value
    };
    enum {
      value = tmp && Check_integral<Rest...>::value
    };
    static_assert(value, "*** ERROR *** Non-integral type parameter found.");
  };

  //! Partial template specialization that finishes the recursion, or
  //specialization used by vectors.
  template <typename last_type> struct Check_integral<last_type> {
    typedef last_type pack_type;
    enum {
      value = std::is_integral<last_type>::value
    };
  };

public:
  //! Indexed access through operator()
  template <typename... Args> reference operator()(Args... params) {

    // check that the number of parameters corresponds to the size of the array
    static_assert(
        sizeof...(Args) == k,
        "*** ERROR *** Number of parameters does not match array rank.");

    typedef typename Check_integral<Args...>::pack_type pack_type;

    // unpack parameters
    pack_type indices[] = { params... };

    // return reference
    return data_[index(indices)];
  }

  //! Indexed access through operator() for constant tensors
  template <typename... Args> value_type operator()(Args... params) const {

    // check that the number of parameters corresponds to the size of the array
    static_assert(
        sizeof...(Args) == k,
        "*** ERROR *** Number of parameters does not match array rank.");

    typedef typename Check_integral<Args...>::pack_type pack_type;

    // unpack parameters
    pack_type indices[] = { params... };

    // return reference
    return data_[index(indices)];
  }

  typedef typename Array_proxy_traits<k, Array>::reference proxy_reference;
  typedef typename Array_proxy_traits<k, Array>::value_type proxy_value;

  //! Indexed access through operator[]
  proxy_reference operator[](size_t i) {
    return Array_proxy_traits<k, Array>::get_reference(*this, i);
  }

  //! Indexed access through operator[] for constant tensors
  proxy_value operator[](size_t i) const {
    return Array_proxy_traits<k, Array>::value(*this, i);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // compound assignment operators

  //! Multiplication compound assignment operator
  Array &operator*=(value_type s) {
    cblas_scal(size(), s, data_, 1);
    return *this;
  }

  //! Division compound assignment operator
  Array &operator/=(value_type s) {
    cblas_scal(size(), value_type(1) / s, data_, 1);
    return *this;
  }

  //! Summation compound assignment operator
  Array &operator+=(const Array &b) {

    // check dimensions
    for (int i = 0; i < k; ++i)
      assert(n_[i] == b.n_[i]);

    // call blas routine to add the arrays
    cblas_axpy(size(), 1.0, b.data_, 1, data_, 1);
    // NOTE: the 1.0 is the factor by which v is scaled
    return *this;
  }

  //! Subtraction compound assignment operator
  Array &operator-=(const Array &b) {

    // check dimensions
    for (int i = 0; i < k; ++i)
      assert(n_[i] == b.n_[i]);

    // call blas routine to add the arrays
    cblas_axpy(size(), -1.0, b.data_, 1, data_, 1);
    return *this;
  }

  //! Helper function used to compute the stride of iterators
  size_t stride(size_t dim) const {

    size_t s = 1;
    for (int j = 0; j < dim; ++j)
      s *= n_[j];
    return s;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // iterator functions

  //! Iterator begin
  iterator begin() { return iterator(data_); }

  //! Iterator begin for constant tensors
  const_iterator begin() const { return const_iterator(data_); }

  //! Iterator end
  iterator end() { return iterator(data_ + size()); }

  //! Iterator end for constant tensors
  const_iterator end() const { return const_iterator(data_ + size()); }

  //! Reverse iterator begin
  reverse_iterator rbegin() { return reverse_iterator(end()); }

  //! Reverse iterator begin for constante tensors
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  //! Reverse iterator end
  reverse_iterator rend() { return reverse_iterator(begin()); }

  //! Reverse iterator end for constante tensors
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  ////////////////////////////////////////////////////////////////////////////////
  // dimensional iterator functions

  template <int d> using diterator = Array_iterator<value_type, pointer, d>;

  //! Dimensional iterator begin
  template <int d> diterator<d> dbegin() {
    return diterator<d>(data_, stride(d));
  }

  //! Dimensional iterator end
  template <int d> diterator<d> dend() {
    size_t s = stride(d);
    return diterator<d>(data_ + stride(d + 1), s);
  }

  //! Dimensional iterator begin
  template <int d, typename iterator> diterator<d> dbegin(iterator it) {
    return diterator<d>(&*it, stride(d));
  }

  //! Dimensional iterator end
  template <int d, typename iterator> diterator<d> dend(iterator it) {
    size_t s = stride(d);
    return diterator<d>(&*it + stride(d + 1), s);
  }

  //! Dimensional iterator begin for constant tensors
  template <int d> diterator<d> dbegin() const {
    return diterator<d>(data_, stride(d));
  }

  //! Dimensional iterator end for constant tensors
  template <int d> diterator<d> dend() const {
    size_t s = stride(d);
    return diterator<d>(data_ + stride(d + 1), s);
  }

  //! Dimensional iterator begin for constant tensors
  template <int d, typename iterator> diterator<d> dbegin(iterator it) const {
    return diterator<d>(&*it, stride(d));
  }

  //! Dimensional iterator end for constant tensors
  template <int d, typename iterator> diterator<d> dend(iterator it) const {
    size_t s = stride(d);
    return diterator<d>(&*it + stride(d + 1), s);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // friend classes and functions

  friend class Array_traits<k, T, Array>;

  template <int dim, class array_type> friend struct Array_proxy;

  friend Array_proxy_traits<k, Array>;
  friend class ApAdd;
  friend class ApSub;
  friend class ApMul;

  //! Standard output
  friend std::ostream &operator<<(std::ostream &os, const Array &a) {
    if (a.size() == 0) {
      os << "Empty array" << endl;
      return os;
    }
    if (a.wrapped_)
      os << "Wrapped ";
    Print<k>::print(os, a.n_, a.data_);
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////
// implementation of array functions

// copy constructor
template <int k, typename T>
Array<k, T>::Array(const Array<k, T> &a)
    : data_(nullptr), wrapped_() {

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
Array<k, T>::Array(Array<k, T> &&src)
    : data_(nullptr), wrapped_() {

  std::copy_n(src.n_, k, n_);
  data_ = src.data_;
  wrapped_ = src.wrapped_;

  std::fill_n(src.n_, k, 0);
  src.data_ = nullptr;
  src.wrapped_ = false;
}

// assignment operator
template <int k, typename T>
Array<k, T> &Array<k, T>::operator=(const Array<k, T> &src) {

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
Array<k, T> &Array<k, T>::operator=(Array<k, T> &&src) {

  if (this != &src) {

    if (this != &src) {

      if (!wrapped_)
        delete[] data_;

      std::copy_n(src.n_, k, n_);
      wrapped_ = src.wrapped_;
      data_ = src.data_;

      // set src to default
      src.data_ = nullptr;
      src.wrapped_ = false;
      std::fill_n(src.n_, k, 0);
    }
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
// alias templates

//! Alias template for vectors
template <class U> using vector_type = array::Array<1, U>;

//! Alias template for matrices
template <class U> using matrix_type = array::Array<2, U>;

//! Alias template for fourth-order tensors
template <class U> using tensor_type = array::Array<4, U>;

////////////////////////////////////////////////////////////////////////////////
// array proxy classes for indexed access through operator[]

//! Proxy class template used to provide the Array with indexed access through
//operator[].
template <int d, class Array> struct Array_proxy {

  typedef const Array_proxy<d - 1, Array> value_type;
  typedef Array_proxy<d - 1, Array> reference;

  //! Array constructor
  /*! The index is taken as the first component of operator[]
   * and no further update is needed
   */
  explicit Array_proxy(const Array &a, size_t i) : a_(a), i_(i), s_(a.n_[0]) {}

  //! Parameter constructor
  template <int c>
  Array_proxy(const Array_proxy<c, Array> &a, size_t i)
      : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::rank() - c]) {}

  reference operator[](size_t i) { return reference(*this, i); }

  value_type operator[](size_t i) const { return value_type(*this, i); }

  const Array &a_;
  size_t i_, s_;
};

//! Proxy class template partial template specialization for last index.
template <class Array> struct Array_proxy<1, Array> {

  typedef typename Array::reference reference;
  typedef typename Array::value_type value_type;

  //! Array constructor
  /*! The index is taken as the first component of operator[]
   * and no further update is needed
   */
  explicit Array_proxy(const Array &a, size_t i) : a_(a), i_(i), s_(a.n_[0]) {}

  //! Parameter constructor
  template <int c>
  Array_proxy(const Array_proxy<c, Array> &a, size_t i)
      : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Array::rank() - c]) {}

  reference operator[](size_t i) { return a_.data_[i_ + i * s_]; }

  value_type operator[](size_t i) const { return a_.data_[i_ + i * s_]; }

private:
  const Array &a_;
  size_t i_, s_;
};

//! Print template partial specialization for vectors
template <> struct Print<1> {

  template <typename value_type>
  static std::ostream &print(std::ostream &os, const size_t size[],
                             const value_type *data) {

    const size_t m = size[0];
    os << "Array<1> (" << m << ")" << endl;
    for (size_t i = 0; i < m; ++i)
      os << ' ' << data[i] << '\n';
    return os;
  }
};

//! Print template partial specialization for matrices
template <> struct Print<2> {

  template <typename value_type>
  static std::ostream &print(std::ostream &os, const size_t size[],
                             const value_type *data) {

    const size_t m = size[0];
    const size_t n = size[1];

    os << "Array<2> (" << m << "x" << n << ")" << endl;
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j)
        os << " " << data[i + j * m];
      cout << endl;
    }
    return os;
  }
};

//! Print class template that is used for tensors of order greater than 2
template <int d> struct Print {

  template <typename value_type>
  static std::ostream &print(std::ostream &os, const size_t size[],
                             const value_type *data) {

    size_t s = 1;
    for (int i = 0; i < d - 1; ++i)
      s *= size[i];

    for (size_t i = 0; i < size[d - 1]; ++i) {
      os << "Dim " << d << ": " << i << ", ";
      Print<d - 1>::print(os, size, data + i * s);
    }
    return os;
  }
};

__END_ARRAY_NAMESPACE__

#endif /* ARRAY_HPP */

////////////////////////////////////////////////////////////////////////////////
// documentation

/*!

 \verbatim
                                                                                               
    _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/  _/_/  _/  _/_/    _/_/_/  _/    _/   
 _/        _/    _/  _/    _/  _/_/_/_/_/  _/    _/  _/_/      _/_/      _/    _/  _/    _/    
_/        _/    _/  _/    _/              _/    _/  _/        _/        _/    _/  _/    _/     
 _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/        _/          _/_/_/    _/_/_/      
        _/        _/                                                                  _/       
       _/        _/                                                              _/_/          
 
 \endverbatim


 * \mainpage A C++ interface to the BLAS library using arbitrary-rank arrays
 * \author Alejandro Marcos Aragón, Ph.D.
 * \n Email: alejandro.aragon@fulbrightmail.org
 *
 * \date 06/05/2013
 * \note Preprint  <a href="http://arxiv.org/abs/1209.1003">arXiv:1209.1003</a>


 \section intro_sec Introduction


 The \c cpp-array project aims at creating a library of algebraic objects
 that interfaces to the \c BLAS set of functions to maintain high performance.
 The main aim of the library is to provide the end user with an easy to
 use syntax, where mathematical expressions can be used and the library
 takes care of calling the \c BLAS routines behind the scenes. This is done
 through the use of expression templates and features provided by the new
 standard library (C++11) so performance is not affected at all.
 Because \c cpp-array is an interface to the \c BLAS set of routines, a working
 \c BLAS implmenentation is needed.


 \section prereq Prerequisites

 The compilation of \c cpp-array requires the following packages:

 - \c CMake build system,
 - \c GNU \c make,
 - a C++ compiler that supports the C++11 set of requirements (tested
 on gcc 4.7 and clang 4.2, or newer versions of them),
 - An implementation of the \c BLAS set of routines (Fortran compiler needed) or
 its C interface (\c CBLAS),
 - \c Doxygen (for the documentation).

 \section config Configuration

 Unzip the \c cpp-array .zip package file. Change to the unzipped directory
using the command \c cd. The library can be configured in three different ways
from the command line. If a Fortran compiler is given by using the \c fc
parameter, the code tries to find a working Fortran BLAS implementation:
 \verbatim $ make config fc=gfortran \endverbatim

 If the user wants to use the Nvidia CUBLAS implementation instead to run the
code in GPUs, the library can be configured as follows:
 \verbatim $ make config cuda=1 \endverbatim

 If none of these parameters are given, the configure program will try to find a
C interface to the BLAS library:
 \verbatim $ make config \endverbatim

 The following table lists all the parameters that can be used to configure the
library:

 option               | default       | description
 ----------------     | ------------- | ------------
 \c cxx=[compiler]    | c++           | The C++ compiler to use
 \c fc=[compiler]     | None          | The Fortran compiler to use
 \c cuda=[bool]       | false         | Enable GPU computing
 \c prefix=[path]     | /usr/local    | Installation path
 \c doc=[bool]        | true          | Build the documentation
 \c latex=[bool]      | false         | Enable LaTeX documentation
 \c build=[string]    | None          | Build type (Debug, Release, etc.)


 \note There is no default as the library can be built without Fortran support.


 The variables resulting from the configuration can be edited direcly by typing
 \verbatim $ make edit_cache \endverbatim

 This may prove useful, \e e.g., in the case \c CMake fails to find the \c CBLAS
 library because it is located on a non-standard directory.

 \section compil Compilation


 To compile the library once the configuration has taken place, type
 \verbatim $ make \endverbatim

 All programs are stored within the directory build/\<architecture\>/, where
 \<architecture\> is machine dependent (\e e.g., Linux-x86_64, or
Darwin-x86_64).

 \section install Installation


 Install the library by typing
 \verbatim $ make install \endverbatim

 \section other Other

 To remove the installed files, type
 \verbatim $ make uninstall \endverbatim

 To remove all object files while retaining the configuration, type
 \verbatim $ make clean \endverbatim

 To remove the build directory, type
 \verbatim $ make distclean \endverbatim

 The documentation for the library can be built by running
 \verbatim $ make doc \endverbatim
 assuming that \c Doxygen is installed in the system. The documentation is
 also generated within the directory build/\<architecture\>/.

 The \c examples/ folder shows how to use the library with a set of examples.
 Check the source files in this folder for a correct usage of the library.
 To build the examples, type
 \verbatim $ make examples \endverbatim

 To run the examples, change to the directory build/\<architecture\>/examples
 and execute the corresponding program.

 The tests in the \c tests/ directory are run to make sure that the library is
 built correctly. If any of the tests fails, check the difference between the
 .\c lastout and the .\c verified files for that particular test in order to
find
 the problem. The tests are executed by the \c CTest framework, by typing
 \verbatim $ make check \endverbatim

 \section devel For developers


 The \c CMake build system can be accessed directly to have better control on
the
 build options. By running the command
 \verbatim $ make edit_cache \endverbatim
 the user can customize many more options than those presented above.

 */
