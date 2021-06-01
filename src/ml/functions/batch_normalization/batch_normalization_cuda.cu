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

#include "batch_normalization.h"
#include "batch_normalization_cuda.h"

__global__ void math21_ml_batchnormalization_backward_mu_fast_cuda_kernel(const float *dX_hat, const float *variance,
                                                                          int mini_batch_size, int features_size,
                                                                          int in_class_size, float *dmu) {
    __shared__ float local[MATH21_CUDA_BLOCK_SIZE];

    int id = threadIdx.x;
    local[id] = 0;

    int ifeature = blockIdx.x;

    int imb, imember;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; imember += MATH21_CUDA_BLOCK_SIZE) {
            int index = (imb * features_size + ifeature) * in_class_size + imember + id;
            local[id] += (imember + id < in_class_size) ? dX_hat[index] : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        dmu[ifeature] = 0;
        int i;
        for (i = 0; i < MATH21_CUDA_BLOCK_SIZE; ++i) {
            dmu[ifeature] += local[i];
        }
        dmu[ifeature] *= (-1.f / sqrtf(variance[ifeature] + .00001f));
    }
}

void math21_ml_batchnormalization_backward_mu_fast_cuda(const float *dX_hat, const float *variance, int mini_batch_size,
                                                        int features_size, int in_class_size, float *dmu) {
    math21_ml_batchnormalization_backward_mu_fast_cuda_kernel << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                                                                  (dX_hat, variance, mini_batch_size, features_size, in_class_size, dmu);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_ml_batchnormalization_backward_mu_cuda_kernel(const float *dX_hat, const float *variance, int mini_batch_size,
                                                     int features_size, int in_class_size, float *dmu) {
    int ifeature = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (ifeature >= features_size) return;
    int imb, imember;
    dmu[ifeature] = 0;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; ++imember) {
            int index = (imb * features_size + ifeature) * in_class_size + imember;
            dmu[ifeature] += dX_hat[index];
        }
    }
    dmu[ifeature] *= (-1.f / sqrtf(variance[ifeature] + .00001f));
}

void math21_ml_batchnormalization_backward_mu_cuda(const float *dX_hat, const float *variance, int mini_batch_size,
                                                   int features_size, int in_class_size, float *dmu) {
    math21_ml_batchnormalization_backward_mu_cuda_kernel << < math21_cuda_gridsize(features_size),
            MATH21_CUDA_BLOCK_SIZE >> > (dX_hat, variance, mini_batch_size, features_size, in_class_size, dmu);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_ml_batchnormalization_backward_sigma_square_cuda_kernel(float *x, float *delta, float *mean, float *variance,
                                                               int mini_batch_size, int features_size,
                                                               int in_class_size, float *variance_delta) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= features_size) return;
    int j, k;
    variance_delta[i] = 0;
    for (j = 0; j < mini_batch_size; ++j) {
        for (k = 0; k < in_class_size; ++k) {
            int index = j * features_size * in_class_size + i * in_class_size + k;
            variance_delta[i] += delta[index] * (x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * powf(variance[i] + .00001f, (float) (-3.f / 2.f));
}

__global__ void math21_ml_batchnormalization_backward_sigma_square_fast_cuda_kernel(const float *x, const float *delta,
                                                                                    const float *mean,
                                                                                    const float *variance,
                                                                                    int mini_batch_size,
                                                                                    int features_size,
                                                                                    int in_class_size,
                                                                                    float *variance_delta) {
    const int threads = MATH21_CUDA_BLOCK_SIZE;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int ifeature = blockIdx.x;

    int i, j;
    for (j = 0; j < mini_batch_size; ++j) {
        for (i = 0; i < in_class_size; i += threads) {
            int index = j * in_class_size * features_size + ifeature * in_class_size + i + id;

            local[id] += (i + id < in_class_size) ? delta[index] * (x[index] - mean[ifeature]) : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        variance_delta[ifeature] = 0;
        for (i = 0; i < threads; ++i) {
            variance_delta[ifeature] += local[i];
        }
        variance_delta[ifeature] *= -.5f * powf(variance[ifeature] + .00001f, (float) (-3.f / 2.f));
    }
}

void math21_ml_batchnormalization_backward_sigma_square_fast_cuda(const float *X, const float *dX_hat, const float *mu,
                                                                  const float *variance, int mini_batch_size,
                                                                  int features_size, int in_class_size,
                                                                  float *dvariance) {
    math21_ml_batchnormalization_backward_sigma_square_fast_cuda_kernel << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                                                                            (X, dX_hat, mu, variance, mini_batch_size, features_size, in_class_size, dvariance);
    math21_cuda_check_error(cudaPeekAtLastError());
}


__global__ void
math21_ml_batchnormalization_backward_input_cuda_kernel(int N, const float *x, const float *mean, const float *variance,
                                                        const float *dmu, const float *variance_delta,
                                                        int mini_batch_size, int features_size, int in_class_size,
                                                        float *delta) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index / in_class_size) % features_size;

    delta[index] = delta[index] * 1.f / (sqrtf(variance[f] + .00001f)) +
                   variance_delta[f] * 2.f * (x[index] - mean[f]) / (in_class_size * mini_batch_size) +
                   dmu[f] / (in_class_size * mini_batch_size);
}

void math21_ml_batchnormalization_backward_input_cuda(const float *X, const float *mu, const float *variance,
                                                      const float *dmu, const float *dvariance, int mini_batch_size,
                                                      int features_size, int in_class_size, float *dX_hat) {
    size_t N = mini_batch_size * features_size * in_class_size;
    math21_ml_batchnormalization_backward_input_cuda_kernel << < math21_cuda_gridsize(N), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                          (N, X, mu, variance, dmu, dvariance, mini_batch_size, features_size, in_class_size, dX_hat);
    math21_cuda_check_error(cudaPeekAtLastError());
}