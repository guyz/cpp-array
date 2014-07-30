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

/*! \file lapack_impl.hpp
 *
 * \brief This file contains the implementation of the function calls when the
 * library is configured to work with LAPACK.
 */

#ifndef LAPACK_IMPL_HPP
#define LAPACK_IMPL_HPP

#ifdef HAVE_LAPACK
#define CPPARRAY_CLAPACK(name, NAME) CPPARRAY_FC_GLOBAL(name, NAME)
#include "fortran_mangling.hh"
#elif defined(CLAPACK_APPLE)
#define CPPARRAY_CLAPACK(name, NAME) name##_
#elif defined(CLAPACK_MKL)
#define CPPARRAY_CLAPACK(name, NAME) c##name##_
#else
#define CPPARRAY_CLAPACK(name, NAME) clapack_##name
#endif

__BEGIN_ARRAY_NAMESPACE__

extern "C" {

// LU decomoposition of a general matrix
void CPPARRAY_CLAPACK(sgetrf, SGETRF)(int *M, int *N, float *A, int *lda,
                                      int *IPIV, int *INFO);
void CPPARRAY_CLAPACK(dgetrf, DGETRF)(int *M, int *N, double *A, int *lda,
                                      int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void CPPARRAY_CLAPACK(sgetri, SGETRI)(int *N, float *A, int *lda, int *IPIV,
                                      float *WORK, int *lwork, int *INFO);
void CPPARRAY_CLAPACK(dgetri, DGETRI)(int *N, double *A, int *lda, int *IPIV,
                                      double *WORK, int *lwork, int *INFO);
}

// LU decomoposition of a general matrix
static void MAY_NOT_BE_USED
lapack_Xgetrf(int *M, int *N, float *A, int *lda, int *IPIV, int *INFO) {
  CPPARRAY_CLAPACK(sgetrf, SGETRF)(M, N, A, lda, IPIV, INFO);
}

static void MAY_NOT_BE_USED
lapack_Xgetrf(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO) {
  CPPARRAY_CLAPACK(dgetrf, DGETRF)(M, N, A, lda, IPIV, INFO);
}

template <typename T>
static void lapack_getrf(int M, int N, T *A, int lda, int *IPIV, int *INFO) {
  lapack_Xgetrf(&M, &N, A, &lda, IPIV, INFO);
}

// generate inverse of a matrix given its LU decomposition
static void MAY_NOT_BE_USED lapack_Xgetri(int *N, float *A, int *lda, int *IPIV,
                                          float *WORK, int *lwork, int *INFO) {
  CPPARRAY_CLAPACK(sgetri, SGETRI)(N, A, lda, IPIV, WORK, lwork, INFO);
}

static void MAY_NOT_BE_USED lapack_Xgetri(int *N, double *A, int *lda,
                                          int *IPIV, double *WORK, int *lwork,
                                          int *INFO) {
  CPPARRAY_CLAPACK(dgetri, DGETRI)(N, A, lda, IPIV, WORK, lwork, INFO);
}

template <typename T>
static void lapack_getri(int N, T *A, int lda, int *IPIV, T *WORK, int lwork,
                         int *INFO) {
  lapack_Xgetri(&N, A, &lda, IPIV, WORK, &lwork, INFO);
}

__END_ARRAY_NAMESPACE__

#endif /* LAPACK_IMPL_HPP */
