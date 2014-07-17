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

/*! \file blas_impl.hpp
 *
 * \brief This file contains the implementation of the function calls when the
 * library is configured to work with a Fortran BLAS library. The final function
 * calls take into account the mangling of symbols of the BLAS library.
 */

#ifndef cpp_array_blas_impl_hpp
#define cpp_array_blas_impl_hpp

#include "fortran_mangling.hh"

__BEGIN_ARRAY_NAMESPACE__

//! Macro used to set the 'transpose' flag
#define CblasTrans 'T'
//! Macro used to set the 'no transpose' flag
#define CblasNoTrans 'N'

extern "C" {

// level 1 blas xNRM2

/*! \brief Level 1 blas used to compute the 2-norm of a vector of single
 * precision type taking into account the Fortran mangling
 */
float CPPARRAY_FC_GLOBAL(snrm2, SNRM2)(int *, float *, int *);

/*! \brief Level 1 blas used to compute the 2-norm of a vector of double
 * precision type taking into account the Fortran mangling
 */
double CPPARRAY_FC_GLOBAL(dnrm2, DNRM2)(int *, double *, int *);

// level 1 blas xSCAL function: x <- alpha*x

/*! \brief Level 1 blas used to scale a vector of single precision type taking
 * into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(sscal, SSCAL)(int *, float *, float *, int *);

/*! \brief Level 1 blas used to scale a vector of double precision type taking
 * into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(dscal, DSCAL)(int *, double *, double *, int *);

// level 1 blas xAXPY function: y <- alpha*x + y

/*! \brief Level 1 blas used to scale and add a vector of single precision
 * type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(saxpy, SAXPY)(int *, float *, float *, int *, float *,
                                      int *);

/*! \brief Level 1 blas used to scale and add a vector of double precision
 * type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(daxpy, DAXPY)(int *, double *, double *, int *,
                                      double *, int *);

// level 1 blas xDOT function: dot <- x'*y

/*! \brief Level 1 blas function used to compute the dot product between two
 * vectors of single precision type taking into account the Fortran mangling
 */
float CPPARRAY_FC_GLOBAL(sdot, SDOT)(int *, float *, int *, float *, int *);

/*! \brief Level 1 blas function used to compute the dot product between two
 * vectors of double precision type taking into account the Fortran mangling
 */
double CPPARRAY_FC_GLOBAL(ddot, DDOT)(int *, double *, int *, double *, int *);

// level 2 blas xGER function: A <- alpha*x*y' + A

/*! \brief Level 2 blas function used to compute the outer product of two
 * vectors of single precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(sger, SGER)(int *, int *, float *, float *, int *,
                                    float *, int *, float *, int *);

/*! \brief Level 2 blas function used to compute the outer product of two
 * vectors of double precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(dger, DGER)(int *, int *, double *, double *, int *,
                                    double *, int *, double *, int *);

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y

/*! \brief Level 2 blas function used to multiply a matrix by a vector of
 * single precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(sgemv, SGEMV)(char *, int *, int *, float *, float *,
                                      int *, float *, int *, float *, float *,
                                      int *);

/*! \brief Level 2 blas function used to multiply a matrix by a vector of
 * double precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(dgemv, DGEMV)(char *, int *, int *, double *, double *,
                                      int *, double *, int *, double *,
                                      double *, int *);

// level 3 blas xGEMM function: C <- alpha*op(A)*op(B) = beta*C, op(X) = X, X'

/*! \brief Level 3 blas function used to multiply two matrices of single
 * precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(sgemm, SGEMM)(char *, char *, int *, int *, int *,
                                      float *, float *, int *, float *, int *,
                                      float *, float *, int *);

/*! \brief Level 3 blas function used to multiply two matrices of double
 * precision type taking into account the Fortran mangling
 */
void CPPARRAY_FC_GLOBAL(dgemm, DGEMM)(char *, char *, int *, int *, int *,
                                      double *, double *, int *, double *,
                                      int *, double *, double *, int *);
}

// level 1 blas xNRM2

/*! \brief Level 1 blas concrete function used to compute the 2-norm of a vector
 * of single precision type
 */
static float MAY_NOT_BE_USED cblas_Xnrm2(int *N, float *X, int *incX) {
  return CPPARRAY_FC_GLOBAL(snrm2, SNRM2)(N, X, incX);
}

/*! \brief Level 1 blas concrete function used to compute the 2-norm of a vector
 * of double precision type
 */
static double MAY_NOT_BE_USED cblas_Xnrm2(int *N, double *X, int *incX) {
  return CPPARRAY_FC_GLOBAL(dnrm2, DNRM2)(N, X, incX);
}

// level 1 blas xNRM2 function: nrm2 <- |x|_2
/*! \brief Level 1 blas template function used to compute the 2-norm of a vector
 *
 * This funciton is used to evaluate \f$ r \leftarrow \left\Vert x \right\Vert_2
 * \f$. The funciton is a function template, and the implementation calls the
 * function \c cblas_Xnrm2 for the correct type.
 *
 * \tparam T - Template parameter that defines the type of elements in the
 * vector
 * \param N - The size of vector \f$ x \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 */
template <typename T> static T cblas_nrm2(int N, T *x, int incX) {
  return cblas_Xnrm2(&N, x, &incX);
}

// level 1 blas xSCAL function: x <- alpha*x

/*! \brief Level 1 blas concrete function used to scale a vector of single
 * precision type
 */
static void MAY_NOT_BE_USED
cblas_Xscal(int *N, float *alpha, float *X, int *incX) {
  CPPARRAY_FC_GLOBAL(sscal, SSCAL)(N, alpha, X, incX);
}

/*! \brief Level 1 blas concrete function used to scale a vector of double
 * precision type
 */
static void MAY_NOT_BE_USED
cblas_Xscal(int *N, double *alpha, double *X, int *incX) {
  CPPARRAY_FC_GLOBAL(dscal, DSCAL)(N, alpha, X, incX);
}

// level 1 blas xSCAL function: x <- alpha*x
/*! \brief Level 1 blas template function used to scale a vector
 *
 * This funciton is used to evaluate \f$ x \leftarrow \alpha x \f$. The funciton
 *is a function template, and the implementation calls the
 * function \c cblas_Xscal for the correct type.
 *
 * \tparam T - Template parameter that defines the type of elements in the
 * vector
 * \param N - The size of vector \f$ x \f$
 * \param alpha - Specifies the scalar \f$ \alpha \f$
 * \param x - A one-dimensional array used to store \f$ x \f$
 * \param incX - Increment step used in array \f$ x \f$
 */
template <typename T> static void cblas_scal(int N, T alpha, T *x, int incX) {
  cblas_Xscal(&N, &alpha, x, &incX);
}

// level 1 blas xAXPY function: y <- alpha*x + y

/*! \brief Level 1 blas concrete function used to scale and add a vector of
 * single precision type
 */
static void MAY_NOT_BE_USED
cblas_xaxpy(int *N, float *alpha, float *X, int *incX, float *Y, int *incY) {
  CPPARRAY_FC_GLOBAL(saxpy, SAXPY)(N, alpha, X, incX, Y, incY);
}

/*! \brief Level 1 blas concrete function used to scale and add a vector of
 * double precision type
 */
static void MAY_NOT_BE_USED
cblas_xaxpy(int *N, double *alpha, double *X, int *incX, double *Y, int *incY) {
  CPPARRAY_FC_GLOBAL(daxpy, DAXPY)(N, alpha, X, incX, Y, incY);
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
static void MAY_NOT_BE_USED
cblas_axpy(int N, T alpha, T *x, int incX, T *y, int incY) {
  cblas_xaxpy(&N, &alpha, x, &incX, y, &incY);
}

// level 1 blas xDOT function: dot <- x'*y

/*! \brief Level 1 blas concrete function used to compute the dot product
 * between two vectors of single precision type
 */
static float MAY_NOT_BE_USED
cblas_xdot(int *N, float *X, int *incX, float *Y, int *incY) {
  return CPPARRAY_FC_GLOBAL(sdot, SDOT)(N, X, incX, Y, incY);
}

/*! \brief Level 1 blas concrete function used to compute the dot product
 * between two vectors of double precision type
 */
static double MAY_NOT_BE_USED
cblas_xdot(int *N, double *X, int *incX, double *Y, int *incY) {
  return CPPARRAY_FC_GLOBAL(ddot, DDOT)(N, X, incX, Y, incY);
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
static double MAY_NOT_BE_USED cblas_dot(int N, T *x, int incX, T *y, int incY) {
  return cblas_xdot(&N, x, &incX, y, &incY);
}

// level 2 blas xGER function: A <- alpha*x*y' + A

/*! \brief Level 2 blas concrete function used to compute the outer product of
 * two vectors of single precision type
 */
static void MAY_NOT_BE_USED cblas_xger(int *M, int *N, float *alpha, float *X,
                                       int *incX, float *Y, int *incY, float *A,
                                       int *lda) {
  CPPARRAY_FC_GLOBAL(sger, SGER)(M, N, alpha, X, incX, Y, incY, A, lda);
}

/*! \brief Level 2 blas concrete function used to compute the outer product of
 * two vectors of double precision type
 */
static void MAY_NOT_BE_USED cblas_xger(int *M, int *N, double *alpha, double *X,
                                       int *incX, double *Y, int *incY,
                                       double *A, int *lda) {
  CPPARRAY_FC_GLOBAL(dger, DGER)(M, N, alpha, X, incX, Y, incY, A, lda);
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
static void MAY_NOT_BE_USED cblas_ger(int M, int N, T alpha, T *x, int incX,
                                      T *y, int incY, T *A, int lda) {
  cblas_xger(&M, &N, &alpha, x, &incX, y, &incY, A, &lda);
}

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y

/*! \brief Level 2 blas concrete function used to multiply a matrix by a vector
 * of single precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemv(char *TransA, int *M, int *N, float *alpha, float *A, int *lda,
            float *X, int *incX, float *beta, float *Y, int *incY) {
  CPPARRAY_FC_GLOBAL(sgemv, SGEMV)(TransA, M, N, alpha, A, lda, X, incX, beta,
                                   Y, incY);
}

/*! \brief Level 2 blas concrete function used to multiply a matrix by a vector
 * of double precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemv(char *TransA, int *M, int *N, double *alpha, double *A, int *lda,
            double *X, int *incX, double *beta, double *Y, int *incY) {
  CPPARRAY_FC_GLOBAL(dgemv, DGEMV)(TransA, M, N, alpha, A, lda, X, incX, beta,
                                   Y, incY);
}

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y
/*! \brief Level 2 blas template function used to multiply a matrix by a vector
 *
 * This funciton is used to evaluate \f$ y \leftarrow \alpha \text{op}(A) x +
 *\beta y\f$. The funciton is a function template, and the implementation calls
 *the function \c cblas_xgemv for the correct type.
 *
 * \tparam T - Template parameter that defines the type of the elements involved
 * in the multiplication
 * \param TransA - Specifies the form of \f$ \text{op}(A) \f$ used in the
 * multiplication:
 * CblasNoTrans: \f$\text{op}(A)=A\f$, CblasTrans: \f$\text{op}(A)=A^\top\f$
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
cblas_gemv(char TransA, int M, int N, T alpha, T *A, int lda, T *x, int incX,
           T beta, T *y, int incY) {
  cblas_xgemv(&TransA, &M, &N, &alpha, A, &lda, x, &incX, &beta, y, &incY);
}

/*! \brief Level 3 blas concrete function used to multiply two matrices of
 * single precision type
 */
static void MAY_NOT_BE_USED cblas_xgemm(char *TransA, char *TransB, int *M,
                                        int *N, int *K, float *alpha, float *A,
                                        int *lda, float *B, int *ldb,
                                        float *beta, float *C, int *ldc) {
  CPPARRAY_FC_GLOBAL(sgemm, SGEMM)(TransA, TransB, M, N, K, alpha, A, lda, B,
                                   ldb, beta, C, ldc);
}

/*! \brief Level 3 blas concrete function used to multiply two matrices of
 * double precision type
 */
static void MAY_NOT_BE_USED
cblas_xgemm(char *TransA, char *TransB, int *M, int *N, int *K, double *alpha,
            double *A, int *lda, double *B, int *ldb, double *beta, double *C,
            int *ldc) {
  CPPARRAY_FC_GLOBAL(dgemm, DGEMM)(TransA, TransB, M, N, K, alpha, A, lda, B,
                                   ldb, beta, C, ldc);
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
cblas_gemm(char TransA, char TransB, int M, int N, int K, T alpha, T *A,
           int lda, T *B, int ldb, T beta, T *C, int ldc) {
  cblas_xgemm(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C,
              &ldc);
}

__END_ARRAY_NAMESPACE__

#endif
