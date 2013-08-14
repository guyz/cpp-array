/*
 * Copyright (C) 2013 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@fulbrightmail.org>
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

//! \file cuda_performance.cpp
//
// This file can be used to test the performance of the array in the
// context of matrix-matrix multiplication by using the NVIDIA CUBLAS
// library. Results for double are the default, but the type definition
// could also be changed to float.
//

#include "array-config.hpp"

#include "expr.hpp"


#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>


using std::cout;
using std::endl;
using std::setw;
using namespace std::chrono;

int main(int argc, char **argv) {
  
  typedef double T;
  size_t niter = 10;
  
  array::CUDA::getInstance().initialize(argc, argv);
  array::CUDA::getInstance().info();
  
  size_t memory = array::CUDA::getInstance().memory();
  
  cout<<setw(20)<<"size"<<setw(20)<<"memory (Kb)"<<setw(20)<<"GPU timing"<<setw(20)<<"normal timing"<<endl;
  
  // loop over matrix sizes
  for (size_t m=1;; m *= 2) {
    
    // get allocated size
    size_t size = 3*m*m*sizeof(T);
    cout<<setw(20)<<m<<setw(10)<<size/1024;
    
    if (memory < size) {
      cout<<"\n\n*** INFO *** Operation exceeded GPU memory\nAborting..."<<endl;
      break;
    }
    
    float gpu_time = 0;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point time = high_resolution_clock::now();
    
    for (size_t i=0; i<niter; ++i) {
      
      cout<<"."<<std::flush;
      
      // create matrices
      array::matrix_type<T> A(m,m);
      array::matrix_type<T> B(m,m);
      
      int k = 0;
      int l = 0;
      for (size_t i=0; i<m; ++i)
        for (size_t j=0; j<m; ++j) {
          A(i,j) = ++k;
          B(i,j) = --l;
        }
      
      // timing events
      cudaEvent_t start;
      cudaError_t error = cudaEventCreate(&start);
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to create start event, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
      cudaEvent_t stop;
      error = cudaEventCreate(&stop);
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to create stop event, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
      error = cudaEventRecord(start, NULL);
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to record start event, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      
      // carry out matrix-matrix multiplication
      array::matrix_type<T> C = A*B;
      
      
      // timing events after the operation
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      
      // compute elapsed time
      duration<T> time_array = duration_cast<duration<T>>(t2 - t1);
      time += (t2-t1);
      
      error = cudaEventRecord(stop, NULL);
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to record stop event, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
      error = cudaEventSynchronize(stop);
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to synchronize on the stop event, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
      float msecTotal = 0.0f;
      error = cudaEventElapsedTime(&msecTotal, start, stop);
      
      gpu_time += msecTotal;
      
      if (error != cudaSuccess) {
        cout<<"*** ERROR *** Failed to get time elapsed between events, error code "<<cudaGetErrorString(error)<<endl;
        exit(EXIT_FAILURE);
      }
      
    } // loop over iterations
    
    
    // compute elapsed time
    duration<T> time_array = duration_cast<duration<T>>(time - start);
    
    // print results
    cout<<setw(17)<<gpu_time/niter;
    cout<<" ms"<<setw(18)<<time_array.count()/niter<<" s"<<endl;
    
  } // loop over matrix sizes
  
  return 0;
}
