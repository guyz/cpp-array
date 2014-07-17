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

//! \example cuda_dot_performance.cpp
//
// This file can be used to test the performance of the array in the
// context of the scalar product by using the NVIDIA CUBLAS library.
// Results for double are the default, but the type definition could
// also be changed to float.
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
  typedef steady_clock clock_type;
  
  size_t niter = 10;
  
  array::CUDA::getInstance().initialize(argc, argv);
  array::CUDA::getInstance().info();
  
  size_t memory = array::CUDA::getInstance().memory();
  
  cout<<setw(20)<<"size"<<setw(20)<<"memory (Kb)"<<setw(20)<<"GPU timing"<<setw(20)<<"normal timing"<<endl;
  
  // loop over matrix sizes
  for (size_t m=1;; m *= 2) {
    
    // get allocated size
    size_t size = 2*m*sizeof(T);
    cout<<setw(20)<<m<<setw(10)<<size/1024;
    
    if (memory < size) {
      cout<<"\n\n*** INFO *** Operation exceeded GPU memory\nAborting..."<<endl;
      break;
    }
    
    float gpu_time = 0;
    clock_type::time_point start = clock_type::now();
    clock_type::time_point time = clock_type::now();
    
    for (size_t i=0; i<niter; ++i) {
      
      cout<<"."<<std::flush;
      
      // create matrices
      array::vector_type<T> x(m,1);
      array::vector_type<T> y(m,1);
      
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
      
      clock_type::time_point t1 = clock_type::now();
      
      // carry out matrix-matrix multiplication
      T z = transpose(x)*y;
      z = 0; // to suppress compiler warning: unused variable
      
      
      // timing events after the operation
      clock_type::time_point t2 = clock_type::now();
      
      // compute elapsed time
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
    auto time_array = duration_cast<microseconds>(time - start);
    
    // print results
    cout<<setw(17)<<gpu_time/niter;
    cout<<" ms"<<setw(17)<<time_array.count()/niter<<" μs"<<endl;
    
  } // loop over matrix sizes
  
  return 0;
}
