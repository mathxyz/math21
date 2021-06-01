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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int m21CudaCurrentDevice;

#ifdef MATH21_FLAG_USE_CUDA

void math21_cuda_set_device(int n);

int math21_cuda_get_device();

void math21_cuda_check_error(cudaError_t status);

void math21_cuda_cublas_check_error(cublasStatus_t status);

dim3 math21_cuda_gridsize(size_t n);

float *math21_cuda_vector_create_from_cpuvector(float *x, size_t n);

void math21_cuda_push_array(float *x_gpu, const float *x, size_t n);

void math21_cuda_push_N8_array(NumN8 *x_gpu, const NumN8 *x, size_t n);

void math21_cuda_pull_array(const float *x_gpu, float *x, size_t n);

void math21_cuda_pull_N8_array(const NumN8 *x_gpu, NumN8 *x, size_t n);

cublasHandle_t math21_cuda_blas_handle();

#endif

#ifdef __cplusplus
}
#endif
