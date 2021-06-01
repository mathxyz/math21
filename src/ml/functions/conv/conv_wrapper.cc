/* Copyright 2015 The math21 Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "conv_cpu.h"
#include "conv_cuda.h"
#include "conv_opencl.h"
#include "conv_wrapper.h"

void math21_ml_function_conv_X_to_X_prime_wrapper(PointerFloatInputWrapper X,
                                                  int nch_X, int nr_X, int nc_X,
                                                  int ksize, int stride, int pad, PointerFloatWrapper X_prime) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_conv_X_to_X_prime_cpu(X, nch_X, nr_X, nc_X, ksize, stride, pad, X_prime);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_conv_X_to_X_prime_cuda(X, nch_X, nr_X, nc_X, ksize, stride, pad, X_prime);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_conv_X_to_X_prime_opencl(X, nch_X, nr_X, nc_X, ksize, stride, pad, X_prime);
#endif

}

void
math21_ml_function_conv_binarize_weights_wrapper(PointerFloatInputWrapper weights, int features_size, int size, PointerFloatWrapper binary) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_conv_binarize_weights_cpu(weights, features_size, size, binary);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_conv_binarize_weights_cuda(weights, features_size, size, binary);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_conv_binarize_weights_opencl(weights, features_size, size, binary);
#endif
}

void math21_ml_function_conv_binarize_input_wrapper(PointerFloatInputWrapper input, int n, PointerFloatWrapper binary) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_conv_binarize_input_cpu(input, n, binary);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_conv_binarize_input_cuda(input, n, binary);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_conv_binarize_input_opencl(input, n, binary);
#endif
}

void math21_ml_function_conv_dX_prime_to_dX_wrapper(PointerFloatInputWrapper dX_prime,
                                                    int nch_X, int nr_X, int nc_X,
                                                    int ksize, int stride, int pad, PointerFloatWrapper dX) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_conv_dX_prime_to_dX_cpu(dX_prime, nch_X, nr_X, nc_X,
                                               ksize, stride, pad, dX);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_conv_dX_prime_to_dX_cuda(dX_prime, nch_X, nr_X, nc_X,
                                                ksize, stride, pad, dX);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_conv_dX_prime_to_dX_opencl(dX_prime, nch_X, nr_X, nc_X,
                                                ksize, stride, pad, dX);
#endif
}

void math21_ml_function_conv_smooth_wrapper(mlfunction_conv *l, int size, float rate){
#if defined(MATH21_FLAG_USE_CPU)
    math21_tool_assert(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_conv_smooth_cuda(l, size, rate);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_conv_smooth_opencl(l, size, rate);
#endif
}