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

#include "dropout_cuda.h"

__global__ void math21_ml_function_dropout_forward_cuda_kernel(const float *x, float *y, int size, const float *rand, float prob, float scale) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id < size) y[id] = (rand[id] < prob) ? 0 : scale * x[id];
}

__global__ void math21_ml_function_dropout_backward_cuda_kernel(const float *x, float *y, int size, const float *rand, float prob, float scale) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id < size) {
        if(rand[id] >= prob){
            y[id] += scale * x[id];
        }
    }
}

void math21_ml_function_dropout_forward_cuda(mlfunction_dropout *f, mlfunction_node *finput, int is_train) {
    if (!is_train) return;
    int size = f->inputs * f->batch;
    if (f->i_time_step==0) {
        math21_vector_pr_rand_uniform_01_wrapper(f->rand, size);
    }

    math21_ml_function_dropout_forward_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                            (finput->y, f->y, size, f->rand, f->rate, f->scale);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void math21_ml_function_dropout_backward_cuda(mlfunction_dropout *f, mlfunction_node *finput) {
    if (!finput->dy) return;
    int size = f->inputs * f->batch;

    math21_ml_function_dropout_backward_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                            (f->dy, finput->dy, size, f->rand, f->rate, f->scale);
    math21_cuda_check_error(cudaPeekAtLastError());
}
