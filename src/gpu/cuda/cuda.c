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

#include <stdio.h>
#include <assert.h>
#include "../../matrix/files_c.h"
#include "cuda.h"

// current used cuda device index
int m21CudaCurrentDevice = 0;

#ifdef MATH21_FLAG_USE_CUDA

void math21_cuda_set_device(int n) {
    m21CudaCurrentDevice = n;
    cudaError_t status = cudaSetDevice(n);
    math21_cuda_check_error(status);
}

int math21_cuda_get_device() {
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    math21_cuda_check_error(status);
    return n;
}

void math21_cuda_check_error(cudaError_t status) {
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        snprintf(buffer, 256, "CUDA Error: %s", s);
        math21_error(buffer);
    }
    if (status2 != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        math21_error(buffer);
    }
}

void math21_cuda_cublas_check_error(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char buffer[256];
        snprintf(buffer, 256, "CUDA CUBLAS Error!");
        math21_error(buffer);
    }
}

dim3 math21_cuda_gridsize(size_t n) {
    size_t k = (n - 1) / MATH21_CUDA_BLOCK_SIZE + 1;
    size_t x = k;
    size_t y = 1;
    // 1d to 2d
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * MATH21_CUDA_BLOCK_SIZE) + 1;
    }
    dim3 d = {x, y, 1};
    return d;
}

// deprecate, use math21_vector_create_from_cpuvector_wrapper instead.
float *math21_cuda_vector_create_from_cpuvector(float *x, size_t n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    if (x) {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        math21_cuda_check_error(status);
    } else {
        math21_vector_set_wrapper(n, 0, x_gpu, 1);
    }
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

void math21_cuda_push_array(float *x_gpu, const float *x, size_t n) {
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    math21_cuda_check_error(status);
}

void math21_cuda_push_N8_array(NumN8 *x_gpu, const NumN8 *x, size_t n) {
    size_t size = sizeof(NumN8) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    math21_cuda_check_error(status);
}

void math21_cuda_pull_array(const float *x_gpu, float *x, size_t n) {
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    math21_cuda_check_error(status);
}

void math21_cuda_pull_N8_array(const NumN8 *x_gpu, NumN8 *x, size_t n) {
    size_t size = sizeof(NumN8) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    math21_cuda_check_error(status);
}

cublasHandle_t math21_cuda_blas_handle() {
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = math21_cuda_get_device();
    if (!init[i]) {
        cublasStatus_t status = cublasCreate(&handle[i]);
        math21_cuda_cublas_check_error(status);
        init[i] = 1;
    }
    return handle[i];
}

#endif