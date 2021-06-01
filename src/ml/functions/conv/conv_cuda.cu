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

#include "conv_cuda.h"

// int nr_X_prime = nch_X * ksize * ksize;
// int num_kernels = nch_X * nc_X_prime_1 * nc_X_prime_2;
// X_prime size: (nch_X * nr_K * nc_K ) * (nc_X_prime_1 * nc_X_prime_2)
__global__ void math21_ml_function_conv_X_to_X_prime_cuda_kernel(int num_kernels, const float *X,
                                                                 int nr_X, int nc_X,
                                                                 int ksize, int pad, int stride,
                                                                 int nc_X_prime_1, int nc_X_prime_2,
                                                                 float *X_prime) {
    // by ye
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;
    int ic2 = index % nc_X_prime_2;
    index = index / nc_X_prime_2;
    int ic1 = index % nc_X_prime_1;
    int ich_X = index / nc_X_prime_1;
    int ir_X_abs = ic1 * stride - pad;
    int ic_X_abs = ic2 * stride - pad;
    X_prime += (ich_X * ksize * ksize * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2;
    X += (ich_X * nr_X + ir_X_abs) * nc_X + ic_X_abs;
    for (int ir_K = 0; ir_K < ksize; ++ir_K) {
        for (int ic_K = 0; ic_K < ksize; ++ic_K) {
            int ir_X = ir_X_abs + ir_K;
            int ic_X = ic_X_abs + ic_K;

            // int ir_X = ir_K + ic1 * stride - pad;
            // int ic_X = ic_K + ic2 * stride - pad;
            // X[(ich_X * nr_X + ir_X) * nc_X + ic_X]
            // X[(ich_X * nr_X + ir_K + ic1 * stride - pad) * nc_X + ic_K + ic2 * stride - pad]
            // X[(ich_X * nr_X + ir_K + ir_X_abs) * nc_X + ic_K + ic_X_abs]

            // nr_X_prime = nch_X * ksize * ksize;
            // ir = (ich_X, ir_K, ic_K)
            // ir = ich_X * nr_K * nc_K + ir_K * nc_K + ic_K
            // index_X_prime = (ir * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2
            *X_prime = (ir_X >= 0 && ic_X >= 0 && ir_X < nr_X && ic_X < nc_X) ?
                       X[ir_K * nc_X + ic_K] : 0;

            X_prime += nc_X_prime_1 * nc_X_prime_2;
        }
    }
}

void math21_ml_function_conv_X_to_X_prime_cuda(const float *X,
                                               int nch_X, int nr_X, int nc_X,
                                               int ksize, int stride, int pad, float *X_prime) {
    int nc_X_prime_1 = (nr_X + 2 * pad - ksize) / stride + 1;
    int nc_X_prime_2 = (nc_X + 2 * pad - ksize) / stride + 1;
    int num_kernels = nch_X * nc_X_prime_1 * nc_X_prime_2;
    math21_ml_function_conv_X_to_X_prime_cuda_kernel << <
    math21_cuda_gridsize(num_kernels), MATH21_CUDA_BLOCK_SIZE >> > (
            num_kernels, X, nr_X, nc_X, ksize, pad,
                    stride, nc_X_prime_1,
                    nc_X_prime_2, X_prime);
}

__global__ void
math21_ml_function_conv_binarize_weights_cuda_kernel(const float *weights, int features_size, int size, float *binary) {
    int ifeature = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (ifeature >= features_size) return;
    int i = 0;
    float mean = 0;
    for (i = 0; i < size; ++i) {
        mean += fabsf(weights[ifeature * size + i]);
    }
    mean = mean / size;
    for (i = 0; i < size; ++i) {
        binary[ifeature * size + i] = (weights[ifeature * size + i] > 0) ? mean : -mean;
    }
}

void math21_ml_function_conv_binarize_weights_cuda(const float *weights, int features_size, int size, float *binary) {
    math21_ml_function_conv_binarize_weights_cuda_kernel << < math21_cuda_gridsize(features_size),
            MATH21_CUDA_BLOCK_SIZE >> > (weights, features_size, size, binary);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_ml_function_conv_binarize_input_cuda_kernel(const float *x, int n, float *binary) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void math21_ml_function_conv_binarize_input_cuda(const float *x, int n, float *binary) {
    math21_ml_function_conv_binarize_input_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                     (x, n, binary);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_ml_function_conv_dX_prime_to_dX_cuda_kernel(int num_kernels, const float *dX_prime,
                                                                   int nr_X, int nc_X,
                                                                   int ksize, int pad, int stride,
                                                                   int nc_X_prime_1, int nc_X_prime_2,
                                                                   float *dX) {
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;
    float val = 0;
    int w = index % nc_X + pad;
    int h = (index / nc_X) % nr_X + pad;
    int c = index / (nc_X * nr_X);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, nc_X_prime_2);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, nc_X_prime_1);
    // equivalent implementation
    int offset =
            (c * ksize * ksize + h * ksize + w) * nc_X_prime_1 * nc_X_prime_2;
    int coeff_h_col = (1 - stride * ksize * nc_X_prime_1) * nc_X_prime_2;
    int coeff_w_col = (1 - stride * nc_X_prime_1 * nc_X_prime_2);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            val += dX_prime[offset + h_col * coeff_h_col + w_col * coeff_w_col];
        }
    }
    dX[index] += val;
}

// todo: check code
void math21_ml_function_conv_dX_prime_to_dX_cuda(const float *dX_prime,
                                                 int nch_X, int nr_X, int nc_X,
                                                 int ksize, int stride, int pad, float *dX) {
    // We are going to launch nch_X * nc_X_prime_1 * nc_X_prime_2 kernels, each
    // kernel responsible for copying a single-channel grid.
    int nc_X_prime_1 = (nr_X + 2 * pad - ksize) / stride + 1;
    int nc_X_prime_2 = (nc_X + 2 * pad - ksize) / stride + 1;
    int num_kernels = nch_X * nr_X * nc_X;
    math21_ml_function_conv_dX_prime_to_dX_cuda_kernel << <
    math21_cuda_gridsize(num_kernels), MATH21_CUDA_BLOCK_SIZE >> > (
            num_kernels, dX_prime, nr_X, nc_X,
                    ksize, pad, stride, nc_X_prime_1, nc_X_prime_2, dX);
}

__global__ void
math21_ml_function_conv_smooth_cuda_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size / 2.f);
    int h_offset = -(size / 2.f);

    int out_index = j + w * (i + h * (k + c * b));
    int l, m;
    for (l = 0; l < size; ++l) {
        for (m = 0; m < size; ++m) {
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w * (cur_h + h * (k + b * c));
            int valid = (cur_h >= 0 && cur_h < h &&
                         cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate * (x[index] - x[out_index]) : 0;
        }
    }
}

void math21_ml_function_conv_smooth_cuda(mlfunction_conv *l, int size, float rate) {
    int h = l->out_h;
    int w = l->out_w;
    int c = l->out_c;

    size_t n = h * w * c * l->batch;

    math21_ml_function_conv_smooth_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                             (l->output, n, l->w, l->h, l->c, size, rate, l->delta);
    math21_cuda_check_error(cudaPeekAtLastError());
}


