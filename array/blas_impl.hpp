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

/*! \file blas_impl.hpp
 *
 * \brief This file defines the blas interface.
 */


#ifndef cpp_array_blas_impl_hpp
#define cpp_array_blas_impl_hpp

#include "fortran_mangling.hh"

__BEGIN_ARRAY_NAMESPACE__


#define CblasTrans 'T'
#define CblasNoTrans 'N'

extern "C" {
  
// level 1 blas xNRM2
float CPPARRAY_FC_GLOBAL(snrm2, SNRM2)(int *, float *, int *);

double CPPARRAY_FC_GLOBAL(dnrm2, DNRM2)(int *, double *, int *);


// level 1 blas xSCAL function: x <- alpha*x
void CPPARRAY_FC_GLOBAL(sscal, SSCAL)(int *,  float *, float *, int *);

void CPPARRAY_FC_GLOBAL(dscal, DSCAL)(int *,  double *, double *, int *);


// level 1 blas xAXPY function: y <- alpha*x + y
void CPPARRAY_FC_GLOBAL(saxpy, SAXPY)(int *, float *, float *, int *, float *, int *);

void CPPARRAY_FC_GLOBAL(daxpy, DAXPY)(int *, double *, double *, int *, double *, int *);


// level 1 blas xDOT function: dot <- x'*y
float CPPARRAY_FC_GLOBAL(sdot, SDOT)(int *, float *, int *, float *, int *);

double CPPARRAY_FC_GLOBAL(ddot, DDOT)(int *, double *, int *, double *, int *);


// level 2 blas xGER function: A <- alpha*x*y' + A
void CPPARRAY_FC_GLOBAL(sger, SGER)(int *,    int *, float *,  float *, int *,
                                    float *,  int *, float *,  int *);

void CPPARRAY_FC_GLOBAL(dger, DGER)(int *,    int *, double *, double *, int *,
                                    double *, int *, double *, int *);

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y
void CPPARRAY_FC_GLOBAL(sgemv, SGEMV)(char *, int *, int *, float *, float *, int *,
                                      float *, int *, float *, float *, int *);


void CPPARRAY_FC_GLOBAL(dgemv, DGEMV)(char *, int *, int *, double *, double *, int *,
                                      double *, int *, double *, double *, int *);


// level 3 blas xGEMM function: C <- alpha*op(A)*op(B) = beta*C, op(X) = X, X'
void CPPARRAY_FC_GLOBAL(sgemm, SGEMM)(char *, char *, int *, int *, int *, float *,
                                      float *, int *, float *, int *, float *, float *, int *);


void CPPARRAY_FC_GLOBAL(dgemm, DGEMM)(char *, char *, int *, int *, int *, double *,
                                      double *, int *, double *, int *, double *, double *, int *);
}


// level 1 blas xNRM2
static float MAY_NOT_BE_USED cblas_Xnrm2(int *N, float *X, int *incX) {
	return CPPARRAY_FC_GLOBAL(snrm2, SNRM2)(N, X, incX);
}

static double MAY_NOT_BE_USED cblas_Xnrm2(int *N, double *X, int *incX) {
	return CPPARRAY_FC_GLOBAL(dnrm2, DNRM2)(N, X, incX);
}

template <typename T>
static T cblas_nrm2(int N, T *X, int incX) {
	return cblas_Xnrm2(&N, X, &incX);
}


// level 1 blas xSCAL function: x <- alpha*x
static void MAY_NOT_BE_USED cblas_Xscal(int *N,  float *alpha, float *X, int *incX) {
	CPPARRAY_FC_GLOBAL(sscal, SSCAL)(N, alpha, X, incX);
}

static void MAY_NOT_BE_USED cblas_Xscal(int *N,  double *alpha, double *X, int *incX) {
	CPPARRAY_FC_GLOBAL(dscal, DSCAL)(N, alpha, X, incX);
}

template <typename T>
static void cblas_scal(int N, T alpha, T *X, int incX) {
	cblas_Xscal(&N, &alpha, X, &incX);
}

// level 1 blas xAXPY function: y <- alpha*x + y
static void MAY_NOT_BE_USED cblas_xaxpy(int *N, float *alpha, float *X, int *incX, float *Y, int *incY) {
	CPPARRAY_FC_GLOBAL(saxpy, SAXPY)(N, alpha, X, incX, Y, incY);
}

static void MAY_NOT_BE_USED cblas_xaxpy(int *N, double *alpha, double *X, int *incX, double *Y, int *incY) {
	CPPARRAY_FC_GLOBAL(daxpy, DAXPY)(N, alpha, X, incX, Y, incY);
}

template <typename T>
static void MAY_NOT_BE_USED cblas_axpy(int N, T alpha, T *X, int incX, T *Y, int incY) {
	cblas_xaxpy(&N, &alpha, X, &incX, Y, &incY);
}

// level 1 blas xDOT function: dot <- x'*y
static float MAY_NOT_BE_USED cblas_xdot(int *N, float *X, int *incX, float *Y, int *incY) {
	return CPPARRAY_FC_GLOBAL(sdot, SDOT)(N, X, incX, Y, incY);
}

static double MAY_NOT_BE_USED cblas_xdot(int *N, double *X, int *incX, double *Y, int *incY) {
	return CPPARRAY_FC_GLOBAL(ddot, DDOT)(N, X, incX, Y, incY);
}

template <typename T>
static double MAY_NOT_BE_USED cblas_dot(int N, T *X, int incX, T *Y, int incY) {
	return cblas_xdot(&N, X, &incX, Y, &incY);
}

// level 2 blas xGER function: A <- alpha*x*y' + A
static void MAY_NOT_BE_USED cblas_xger(int *M, int *N, float *alpha, float *X, int *incX,
                                       float *Y,  int *incY, float *A,  int *lda) {
	CPPARRAY_FC_GLOBAL(sger, SGER)(M, N, alpha, X, incX, Y, incY, A, lda);
}

static void MAY_NOT_BE_USED cblas_xger(int *M, int *N, double *alpha, double *X, int *incX,
                                       double *Y,  int *incY, double *A,  int *lda) {
	CPPARRAY_FC_GLOBAL(dger, DGER)(M, N, alpha, X, incX, Y, incY, A, lda);
}

template <typename T>
static void MAY_NOT_BE_USED cblas_ger(int M, int N, T alpha, T *X, int incX,
                                      T *Y, int incY, T *A, int lda) {
	cblas_xger(&M, &N, &alpha, X, &incX, Y, &incY, A, &lda);
}

// level 2 blas xGEMV function: Y <- alpha*A*x + beta*y
static void MAY_NOT_BE_USED cblas_xgemv(char *TransA, int *M, int *N, float *alpha, float *A, int *lda,
                                        float *X, int *incX, float *beta, float *Y, int *incY) {
	CPPARRAY_FC_GLOBAL(sgemv, SGEMV)(TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

static void MAY_NOT_BE_USED cblas_xgemv(char *TransA, int *M, int *N, double *alpha, double *A, int *lda,
                                        double *X, int *incX, double *beta, double *Y, int *incY) {
	CPPARRAY_FC_GLOBAL(dgemv, DGEMV)(TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <typename T>
static void MAY_NOT_BE_USED cblas_gemv(char TransA, int M, int N, T alpha,  T *A, int lda,
                                       T *X, int incX,  T beta,  T *Y,  int incY) {
	cblas_xgemv(&TransA, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
}

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

static void MAY_NOT_BE_USED cblas_xgemm(char *TransA, char *TransB, int *M, int *N, int *K, float *alpha,
                                        float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc) {
	CPPARRAY_FC_GLOBAL(sgemm, SGEMM)(TransA, TransB, M, N, K, alpha, A,
	                                 lda, B, ldb, beta, C, ldc);
}

static void MAY_NOT_BE_USED cblas_xgemm(char *TransA, char *TransB, int *M, int *N, int *K, double *alpha,
                                        double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc) {
	CPPARRAY_FC_GLOBAL(dgemm, DGEMM)(TransA, TransB, M, N, K, alpha, A,
	                                 lda, B, ldb, beta, C, ldc);
}

template <class T>
void MAY_NOT_BE_USED cblas_gemm(char TransA, char TransB, int M, int N, int K, T alpha, T *A,
                                int lda, T *B, int ldb, T beta, T *C, int ldc) {
	cblas_xgemm(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

__END_ARRAY_NAMESPACE__


#endif
