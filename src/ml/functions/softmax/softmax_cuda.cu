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

#include "softmax_cuda.h"

__device__ void math21_ml_function_softmax_cuda_device(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = expf(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

__global__ void math21_ml_function_softmax_tree_cuda_kernel(float *input, int in_class_size, int mini_batch_size, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= in_class_size*mini_batch_size*groups) return;
    int s = id % in_class_size;
    id = id / in_class_size;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*in_class_size;
    int boff = b*stride;
    math21_ml_function_softmax_cuda_device(input + goff + boff + s, group_size[g], temp, in_class_size, output + goff + boff + s);
}

void math21_ml_function_softmax_tree_cuda(float *input, int in_class_size, int mini_batch_size, int stride, float temp, float *output, m21tree hier)
{
    int *tree_groups_size = math21_vector_create_from_cpuvector_int_wrapper(hier.groups, hier.group_size, 1);
    int *tree_groups_offset = math21_vector_create_from_cpuvector_int_wrapper(hier.groups, hier.group_offset, 1);
    int num = in_class_size*mini_batch_size*hier.groups;
    math21_ml_function_softmax_tree_cuda_kernel<<<math21_cuda_gridsize(num), MATH21_CUDA_BLOCK_SIZE>>>(input, in_class_size, mini_batch_size, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    math21_cuda_check_error(cudaPeekAtLastError());
    math21_vector_free_wrapper((float *)tree_groups_size);
    math21_vector_free_wrapper((float *)tree_groups_offset);
}

__global__ void math21_ml_function_softmax_cuda_kernel(float *input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= mini_batch_size*groups) return;
    int b = id / groups;
    int g = id % groups;
    math21_ml_function_softmax_cuda_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

void math21_ml_function_softmax_cuda(float *input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    math21_ml_function_softmax_cuda_kernel<<<math21_cuda_gridsize(mini_batch_size*groups), MATH21_CUDA_BLOCK_SIZE>>>(input, n, mini_batch_size, batch_offset, groups, group_offset, stride, temp, output);
    math21_cuda_check_error(cudaPeekAtLastError());
}


__global__ void math21_ml_function_softmax_x_ent_cuda_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void math21_ml_function_softmax_x_ent_cuda(int n, float *pred, float *truth, float *delta, float *error)
{
    math21_ml_function_softmax_x_ent_cuda_kernel<<<math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE>>>(n, pred, truth, delta, error);
    math21_cuda_check_error(cudaPeekAtLastError());
}