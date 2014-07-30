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

/*! \file blas.hpp
 *
 * \brief This file defines the cublas interface.
 */

#ifndef BLAS_LAPACK_HPP
#define BLAS_LAPACK_HPP

#ifdef __GNUC__
#define MAY_NOT_BE_USED __attribute__((unused))
#else
#define MAY_NOT_BE_USED
#endif

// include appropriate blas implementation file
#ifdef HAVE_CUBLAS_H
#include "cublas_impl.hpp"
#elif defined(HAVE_BLAS_H)
#include "blas_impl.hpp"
#elif defined(HAVE_CBLAS_H)
#include "cblas_impl.hpp"
#else
#error "No blas implementation found."
#endif

// include appropriate lapack implementation file
#if defined(HAVE_LAPACK) || defined(HAVE_CLAPACK)
#include "lapack_impl.hpp"
#endif

#endif /* BLAS_LAPACK_HPP */
