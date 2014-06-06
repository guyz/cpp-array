/*
 * Copyright (©) 2014 EPFL Alejandro M. Aragón
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

/*! \file cblas_impl.hpp
 *
 * \brief This file defines the cblas interface.
 */

#ifndef CBLAS_IMPL_HPP
#define CBLAS_IMPL_HPP

extern "C"  {
#include CBLAS_HEADER
}

__BEGIN_ARRAY_NAMESPACE__


////////////////////////////////////////////////////////////////////////////////
// blas functions

// level 1 blas xNRM2
static float MAY_NOT_BE_USED cblas_Xnrm2(const int N, const float *X, const int incX) {
	return cblas_snrm2(N, X, incX);
}

static double MAY_NOT_BE_USED cblas_Xnrm2(const int N, const double *X, const int incX) {
	return cblas_dnrm2(N, X, incX);
}

template <typename T>
static T cblas_nrm2(const int N, const T *X, const int incX) {
	return cblas_Xnrm2(N, X, incX);
}

// level 1 blas xSCAL function: x <- alpha*x
static void MAY_NOT_BE_USED cblas_Xscal(const int N, const float alpha, float *X, const int incX)
{ cblas_sscal(N, alpha, X, incX); } 

static void MAY_NOT_BE_USED cblas_Xscal(const int N, const double alpha, double *X, const int incX)
{ cblas_dscal(N, alpha, X, incX); }


template <typename T>
static void cblas_scal(const int N, const T alpha, T *X, const int incX)
{ cblas_Xscal(N, alpha, X, incX); }



// level 1 blas xAXPY function: y <- alpha*x + y
static void MAY_NOT_BE_USED cblas_xaxpy(const int N, const float alpha, const float *X,
                        const int incX, float *Y, const int incY)
{ cblas_saxpy(N, alpha, X, incX, Y, incY); }

static void MAY_NOT_BE_USED cblas_xaxpy(const int N, const double alpha, const double *X,
                        const int incX, double *Y, const int incY)
{ cblas_daxpy(N, alpha, X, incX, Y, incY); }


template <typename T>
static void MAY_NOT_BE_USED cblas_axpy(const int N, const T alpha, const T *X,
                       const int incX, T *Y, const int incY)
{ cblas_xaxpy(N, alpha, X, incX, Y, incY); }




// level 1 blas xDOT function: dot <- x'*y
static float MAY_NOT_BE_USED cblas_xdot(const int N, const float  *X, const int incX,
                        const float  *Y, const int incY)
{ return cblas_sdot(N, X, incX, Y, incY); }

static double MAY_NOT_BE_USED cblas_xdot(const int N, const double *X, const int incX,
                         const double *Y, const int incY)
{ return cblas_ddot(N, X, incX, Y, incY); }


template <typename T>
static double MAY_NOT_BE_USED cblas_dot(const int N, const T *X, const int incX,
                        const T *Y, const int incY)
{ return cblas_xdot(N, X, incX, Y, incY); }



// level 2 blas xGER function: A <- alpha*x*y' + A
static void MAY_NOT_BE_USED cblas_xger(const int M, const int N,
                       const double alpha, const double *X, const int incX,
                       const double *Y, const int incY, double *A, const int lda)
{ cblas_dger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda); }

static void MAY_NOT_BE_USED cblas_xger(const int M, const int N,
                       const float alpha, const float *X, const int incX,
                       const float *Y, const int incY, float *A, const int lda)
{ cblas_sger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda); }



template <typename T>
static void MAY_NOT_BE_USED cblas_ger(const int M, const int N,
                      const T alpha, const T *X, const int incX,
                      const T *Y, const int incY, T *A, const int lda)
{ cblas_xger(M, N, alpha, X, incX, Y, incY, A, lda); }





// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y
static void MAY_NOT_BE_USED cblas_xgemv(const enum CBLAS_TRANSPOSE TransA,
                        const int M, const int N,
                        const double alpha,
                        const double *A,
                        const int lda,
                        const double *X,
                        const int incX,
                        const double beta,
                        double *Y,
                        const int incY)
{ cblas_dgemv(CblasColMajor, TransA, M, N,
              alpha, A, lda, X, incX, beta, Y, incY); }

static void MAY_NOT_BE_USED cblas_xgemv(const enum CBLAS_TRANSPOSE TransA,
                        const int M,
                        const int N,
                        const float alpha,
                        const float *A,
                        const int lda,
                        const float *X,
                        const int incX,
                        const float beta,
                        float *Y,
                        const int incY)
{ cblas_sgemv(CblasColMajor, TransA, M, N,
              alpha, A, lda, X, incX, beta, Y, incY); }


template <typename T>
static void MAY_NOT_BE_USED cblas_gemv(const enum CBLAS_TRANSPOSE TransA,
                       const int M,
                       const int N,
                       const T alpha,
                       const T *A,
                       const int lda,
                       const T *X,
                       const int incX,
                       const T beta,
                       T *Y,
                       const int incY)
{ cblas_xgemv(TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY); }



// level 3 blas xGEMM function: C <- alpha*op(A)*op(B) = beta*C, op(X) = X, X'
/*! \f$ C \leftarrow \alpha op(A)op(B) + \beta C \f$
 *
 * Arguments:
 *
 *    transA   specifies the form of (op)A used in the multiplication:
 *             CblasNoTrans -> (op)A = A, CblasTrans -> (op)A = transpose(A)
 *    transB   specifies the form of (op)B used in the multiplication:
 *             CblasNoTrans -> (op)B = B, CblasTrans -> (op)B = transpose(B)
 *    M        the number of rows of the matrix (op)A and of the matrix C
 *    N        the number of columns of the matrix (op)B and of the matrix C
 *    K        the number of columns of the matrix (op)A and the number of rows of the matrix (op)B
 *    alpha    specifies the scalar alpha
 *    A        a two-dimensional array A
 *    lda      the first dimension of array A
 *    B        a two-dimensional array B
 *    ldb      the first dimension of array B
 *    beta     specifies the scalar beta
 *    C        a two-dimensional array
 *    ldc      the first dimension of array C
 */

static void MAY_NOT_BE_USED cblas_xgemm(const enum CBLAS_TRANSPOSE TransA,
                        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const float alpha, const float *A,
                        const int lda, const float *B, const int ldb,
                        const float beta, float *C, const int ldc) {
  
  cblas_sgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
}


static void MAY_NOT_BE_USED cblas_xgemm(const enum CBLAS_TRANSPOSE TransA,
                        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha, const double *A,
                        const int lda, const double *B, const int ldb,
                        const double beta, double *C, const int ldc) {
  
  cblas_dgemm(CblasColMajor, TransA, TransB, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
}

template <class T>
void MAY_NOT_BE_USED cblas_gemm(const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const T alpha, const T *A,
                const int lda, const T *B, const int ldb,
                const T beta, T *C, const int ldc) {
  
  cblas_xgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


__END_ARRAY_NAMESPACE__


#endif /* CBLAS_IMPL_HPP */
