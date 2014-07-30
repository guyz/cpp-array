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

/*! \file cblas_impl.hpp
 *
 * \brief This file contains the implementation of the function calls when the
 * library is configured to work with a C interface to the BLAS library.
 */

#ifndef CBLAS_IMPL_HPP
#define CBLAS_IMPL_HPP

extern "C" {
#include CBLAS_HEADER
}

__BEGIN_ARRAY_NAMESPACE__

////////////////////////////////////////////////////////////////////////////////
// blas functions

// level 1 blas xNRM2
static float MAY_NOT_BE_USED
cblas_Xnrm2(const int N, const float *X, const int incX) {
  return cblas_snrm2(N, X, incX);
}

static double MAY_NOT_BE_USED
cblas_Xnrm2(const int N, const double *X, const int incX) {
  return cblas_dnrm2(N, X, incX);
}

template <typename T>
static T cblas_nrm2(const int N, const T *X, const int incX) {
  return cblas_Xnrm2(N, X, incX);
}

// level 1 blas xASUM

/*! \brief Level 1 blas concrete function used to compute the sum the absolute
 * values of the elements of a vector of single precision type
 */
static float MAY_NOT_BE_USED
cblas_Xasum(const int N, const float *X, const int incX) {
  return cblas_sasum(N, X, incX);
}

/*! \brief Level 1 blas concrete function used to compute the sum the absolute
 * values of the elements of a vector of double precision type
 */
static double MAY_NOT_BE_USED
cblas_Xasum(const int N, const double *X, const int incX) {
  return cblas_dasum(N, X, incX);
}

// level 1 blas xASUM function: asum <- |x|_1
/*! \brief Level 1 blas template function used to compute the sum the absolute
 * values of the elements of a vector
 *
 * This funciton is used to evaluate \f$ r \leftarrow \left\Vert x \right\Vert_1
 * \f$. The funciton is a function template, and the implementation calls the
 * function \c cblas_Xasum for the correct type.
 *
 * \tparam T - Template parameter that defines the type of elements in the
 * vector
 * \param N - The size of vector \f$ x \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in vector \f$ x \f$
 */
template <typename T> static T cblas_asum(int N, T *x, int incX) {
  return cblas_Xasum(N, x, incX);
}

// level 1 blas xSCAL function: x <- alpha*x
static void MAY_NOT_BE_USED
cblas_Xscal(const int N, const float alpha, float *X, const int incX) {
  cblas_sscal(N, alpha, X, incX);
}

static void MAY_NOT_BE_USED
cblas_Xscal(const int N, const double alpha, double *X, const int incX) {
  cblas_dscal(N, alpha, X, incX);
}

template <typename T>
static void cblas_scal(const int N, const T alpha, T *X, const int incX) {
  cblas_Xscal(N, alpha, X, incX);
}

// level 1 blas xAXPY function: y <- alpha*x + y

/*! \brief Level 1 blas concrete function used to scale and add a vector of
 * single precision type
 */
static void MAY_NOT_BE_USED cblas_xaxpy(const int N, const float alpha,
                                        const float *X, const int incX,
                                        float *Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

/*! \brief Level 1 blas concrete function used to scale and add a vector of
 * double precision type
 */
static void MAY_NOT_BE_USED cblas_xaxpy(const int N, const double alpha,
                                        const double *X, const int incX,
                                        double *Y, const int incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

// level 1 blas xAXPY function: Y <- alpha*x + y

/*! \brief Level 1 blas template function used to scale and add a vector
 *
 * This funciton is used to evaluate \f$ y \leftarrow \alpha x + y \f$.
 * The funciton is a function template, and the implementation calls the
 * function \c cblas_xaxpy for the correct type.
 *
 * \tparam T - Template parameter that defines the type of elements in the
 * vectors
 * \param N - The size of vectors \f$ x \f$ and \f$ y \f$
 * \param alpha - Specifies the scalar \f$ \alpha \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 * \param y - A one-dimensional array \f$ y \f$
 * \param incY - Increment step used in array \f$ y \f$
 */

template <typename T>
static void MAY_NOT_BE_USED cblas_axpy(const int N, const T alpha, const T *x,
                                       const int incX, T *y, const int incY) {
  cblas_xaxpy(N, alpha, x, incX, y, incY);
}

// level 1 blas xDOT function: dot <- x'*y

/*! \brief Level 1 blas concrete function used to compute the dot product
 * between two vectors of single precision type
 */
static float MAY_NOT_BE_USED cblas_xdot(const int N, const float *X,
                                        const int incX, const float *Y,
                                        const int incY) {
  return cblas_sdot(N, X, incX, Y, incY);
}

/*! \brief Level 1 blas concrete function used to compute the dot product
 * between two vectors of double precision type
 */
static double MAY_NOT_BE_USED cblas_xdot(const int N, const double *X,
                                         const int incX, const double *Y,
                                         const int incY) {
  return cblas_ddot(N, X, incX, Y, incY);
}

/*! \brief Level 1 blas template function used to compute the dot product
 *between two vectors
 *
 * This funciton is used to evaluate \f$ r \leftarrow x^\top  y  \f$.
 * The funciton is a function template, and the implementation calls the
 * function \c cblas_xdot for the correct type.
 *
 * \tparam T - Template parameter that defines the type of elements in the
 * vectors
 * \param N - The size of vectors \f$ x \f$ and \f$ y \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 * \param y - A one-dimensional array \f$ y \f$
 * \param incY - Increment step used in array \f$ y \f$
 */
template <typename T>
static double MAY_NOT_BE_USED
cblas_dot(const int N, const T *x, const int incX, const T *y, const int incY) {
  return cblas_xdot(N, x, incX, y, incY);
}

// level 2 blas xGER function: A <- alpha*x*y' + A

/*! \brief Level 2 blas concrete function used to compute the outer product of
 * two vectors of single precision type
 */
static void MAY_NOT_BE_USED
cblas_xger(const int M, const int N, const double alpha, const double *X,
           const int incX, const double *Y, const int incY, double *A,
           const int lda) {
  cblas_dger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}

/*! \brief Level 2 blas concrete function used to compute the outer product of
 * two vectors of double precision type
 */
static void MAY_NOT_BE_USED
cblas_xger(const int M, const int N, const float alpha, const float *X,
           const int incX, const float *Y, const int incY, float *A,
           const int lda) {
  cblas_sger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}

// level 2 blas xGER function: Y <- alpha*A*x + beta*y
/*! \brief Level 2 blas template function used to compute the outer product of
 * two vectors
 *
 * This funciton is used to evaluate \f$ A \leftarrow \alpha x y^top + A \f$.
 * The funciton is a function template, and the implementation calls the
 * function \c cblas_xger for the correct type.
 *
 * \tparam T - Template parameter that defines the type of the elements involved
 * in the multiplication
 * \param M - The number of rows of the matrix \f$ A \f$ and the size of vector
 *\f$ x \f$
 * \param N - The number of columns of the matrix \f$ A \f$ and the size of
 *vector \f$ y \f$
 * \param alpha - Specifies the scalar \f$ \alpha \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 * \param y - A one-dimensional array \f$ y \f$
 * \param incY - Increment step used in array \f$ y \f$
 * \param A - A two-dimensional array \f$ A \f$
 * \param lda - The first dimension of array \f$ A \f$
 */
template <typename T>
static void MAY_NOT_BE_USED
cblas_ger(const int M, const int N, const T alpha, const T *x, const int incX,
          const T *y, const int incY, T *A, const int lda) {
  cblas_xger(M, N, alpha, x, incX, y, incY, A, lda);
}

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y

/*! \brief Level 2 blas concrete function used to multiply a matrix by a vector
 * of single precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
            const double alpha, const double *A, const int lda, const double *X,
            const int incX, const double beta, double *Y, const int incY) {
  cblas_dgemv(CblasColMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
              incY);
}

/*! \brief Level 2 blas concrete function used to multiply a matrix by a vector
 * of double precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
            const float alpha, const float *A, const int lda, const float *X,
            const int incX, const float beta, float *Y, const int incY) {
  cblas_sgemv(CblasColMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
              incY);
}

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y
/*! \brief Level 2 blas template function used to multiply a matrix by a vector
 *
 * This funciton is used to evaluate \f$ y \leftarrow \alpha \text{op}(A) x +
 * \beta y\f$. The funciton is a function template, and the implementation calls
 * the function \c cblas_xgemv for the correct type.
 *
 * \tparam T - Template parameter that defines the type of the elements involved
 * in the multiplication
 * \param TransA - Specifies the form of \f$ \text{op}(A) \f$ used in the
 * multiplication:
 * CblasNoTrans: \f$ \text{op}(A) =A \f$, CblasTrans: \f$ \text{op}(A) = A^top
 * \f$
 * \param M - The number of rows of the matrix \f$ \text{op}(A) \f$
 * \param N - The number of columns of the matrix \f$ \text{op}(A) \f$ and the
 * size of vectors \f$ x \f$ and \f$ y \f$
 * \param alpha - Specifies the scalar \f$ \alpha \f$
 * \param A - A two-dimensional array \f$ A \f$
 * \param lda - The first dimension of array \f$ A \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 * \param beta - Specifies the scalar \f$ \beta \f$
 * \param y - A one-dimensional array \f$ y \f$
 * \param incY - Increment step used in array \f$ y \f$
 */
template <typename T>
static void MAY_NOT_BE_USED
cblas_gemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
           const T alpha, const T *A, const int lda, const T *x, const int incX,
           const T beta, T *y, const int incY) {
  cblas_xgemv(TransA, M, N, alpha, A, lda, x, incX, beta, y, incY);
}

// level 3 blas xGEMM function: C <- alpha*op(A)*op(B) = beta*C, op(X) = X, X'

/*! \brief Level 3 blas concrete function used to multiply two matrices of
 * single precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemm(const enum CBLAS_TRANSPOSE TransA,
            const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
            const int K, const float alpha, const float *A, const int lda,
            const float *B, const int ldb, const float beta, float *C,
            const int ldc) {

  cblas_sgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

/*! \brief Level 3 blas concrete function used to multiply two matrices of
 * double precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemm(const enum CBLAS_TRANSPOSE TransA,
            const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
            const int K, const double alpha, const double *A, const int lda,
            const double *B, const int ldb, const double beta, double *C,
            const int ldc) {

  cblas_dgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

// level 3 blas xGEMM function: C <- alpha*op(A)*op(B) = beta*C, op(X) = X, X'
/*! \brief Level 3 blas template function used to multiply two matrices
 *
 * This funciton is used to evaluate \f$ C \leftarrow \alpha
 * \text{op}(A)\text{op}(B) + \beta C \f$. The funciton is a function template,
 * and the implementation calls the function \c cblas_xgemm for the correct
 * type.
 *
 * \tparam T - Template parameter that defines the type of the matrix elements
 * \param TransA - Specifies the form of \f$ \text{op}(A) \f$ used in the
 * multiplication: CblasNoTrans: \f$ \text{op}(A) = A \f$, CblasTrans: \f$
 * \text{op}(A) = A^\top \f$
 * \param TransB - Specifies the form of \f$ \text{op}(B) \f$  used in the
 * multiplication: CblasNoTrans: \f$ \text{op}(B) = B \f$, CblasTrans: \f$
 * \text{op}(B) = B^\top \f$
 * \param M - The number of rows of the matrix \f$ \text{op}(A) \f$ and of the
 * matrix \f$ C \f$
 * \param N - The number of columns of the matrix \f$ \text{op}(B) \f$ and of
 * the matrix \f$ C \f$
 * \param K - The number of columns of the matrix \f$ \text{op}(A) \f$ and the
 * number of rows
 * of the matrix \f$ \text{op}(B) \f$
 * \param alpha - Specifies the scalar \f$ \alpha \f$
 * \param A - A two-dimensional array \f$ A \f$
 * \param lda - The first dimension of array \f$ A \f$
 * \param B - A two-dimensional array \f$ B \f$
 * \param ldb - The first dimension of array \f$ B \f$
 * \param beta - Specifies the scalar \f$ \beta \f$
 * \param C - A two-dimensional array \f$ C \f$
 * \param ldc - The first dimension of array \f$ C \f$
 */
template <class T>
static void MAY_NOT_BE_USED
cblas_gemm(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
           const int M, const int N, const int K, const T alpha, const T *A,
           const int lda, const T *B, const int ldb, const T beta, T *C,
           const int ldc) {

  cblas_xgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

__END_ARRAY_NAMESPACE__

#endif /* CBLAS_IMPL_HPP */
