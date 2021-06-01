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


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "test_c.h"


__global__ void _print_hello_world(char *a, int N) {
    char p[12] = "Hello_CUDA\n";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = p[idx];
    }
}

void math21_cuda_example_test() {
    char *a_h, *a_d;
    const int N = 12;
    size_t size = N * sizeof(char);
    cudaMallocHost(&a_h, size);
    cudaMalloc(&a_d, size);
    for (int i = 0; i < N; i++) {
        a_h[i] = 0;
    }
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    cudaMemcpy(a_d, a_h, size, kind);
    int blocksize = 4;
    // y = k * x + b, b < x
    // k*x >= y => min_k = (y-1)/x + 1, here min_k is the smallest such k.
    int nblock = (N - 1 + blocksize) / blocksize;
    _print_hello_world << < nblock, blocksize >> > (a_d, N);
    kind = cudaMemcpyDeviceToHost;
    cudaMemcpy(a_h, a_d, sizeof(char) * N, kind);
    printf("%s", a_h);
    cudaFreeHost(a_h);
    cudaFree(a_d);
}

//int main(){
//    math21_cuda_example_test();
//    return 0;
//}