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

#pragma once

#include "inner.h"
#include "conv.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_CUDA

void math21_ml_function_conv_X_to_X_prime_cuda(const float *X,
                                                  int nch_X, int nr_X, int nc_X,
                                                  int ksize, int stride, int pad, float *X_prime);

void math21_ml_function_conv_binarize_weights_cuda(const float *weights, int features_size, int size, float *binary);

void math21_ml_function_conv_binarize_input_cuda(const float *input, int n, float *binary);

void math21_ml_function_conv_dX_prime_to_dX_cuda(const float *dX_prime,
                                                int nch_X, int nr_X, int nc_X,
                                                int ksize, int stride, int pad, float *dX);

void math21_ml_function_conv_smooth_cuda(mlfunction_conv *l, int size, float rate);

#endif

#ifdef __cplusplus
}
#endif
