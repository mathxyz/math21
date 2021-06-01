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

#ifdef MATH21_FLAG_USE_CUDA

// https://cnugteren.github.io/tutorial/pages/page10.html
// Replace the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global
#define __placeholder__ __global__
#define __local __shared__

// Replace OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()

// Replace the OpenCL get_xxx_ID with CUDA equivalents
__device__ int get_local_id(int x) {
    return (x == 0) ? threadIdx.x : threadIdx.y;
}

__device__ int get_group_id(int x) {
    return (x == 0) ? blockIdx.x : blockIdx.y;
}

__device__ int get_global_id(int x) {
    return (x == 0) ? blockIdx.x * blockDim.x + threadIdx.x :
           blockIdx.y * blockDim.y + threadIdx.y;
}

// todo: remove
// Add the float8 data-type which is not available natively under CUDA
typedef struct {
    float s0;
    float s1;
    float s2;
    float s3;
    float s4;
    float s5;
    float s6;
    float s7;
} m21float8;


#endif