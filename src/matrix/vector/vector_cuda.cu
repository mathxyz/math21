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

#include "inner.h"
#include "../../gpu/files.h"
#include "../../generic/files_c.h"
#include "vector_cpu.h"
#include "vector_cuda.h"

using namespace math21;

float *math21_vector_deserialize_c_cuda(FILE *f, size_t *n0) {
    float *x = math21_vector_deserialize_c_cpu(f, n0);
    size_t n = *n0;
    if (!x || n <= 0) {
        return 0;
    }
    float *v = math21_vector_create_with_default_value_cuda(n, 0);
    math21_cuda_push_array(v, x, n);
    math21_vector_free_cpu(x);
    return v;
}

float *math21_vector_deserialize_from_file_cuda(const char *name, size_t *n) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        math21_file_error(name);
        return 0;
    }
    float *v = math21_vector_deserialize_c_cuda(f, n);
    fclose(f);
    return v;
}

void math21_vector_serialize_c_cuda(FILE *f, const float *v, size_t n) {
    if (!v || n <= 0) {
        return;
    }
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_cuda_pull_array(v, x, n);
    math21_vector_serialize_c_cpu(f, x, n);
    math21_vector_free_cpu(x);
}

void math21_vector_serialize_to_file_cuda(const char *name, const float *v, size_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        math21_file_error(name);
        return;
    }
    math21_vector_serialize_c_cuda(f, v, n);
    fclose(f);
}

// [from, to)
void math21_vector_save_cuda(const char *name, const float *v, size_t from, size_t to) {
    size_t n = to - from;
    if (!v || n <= 0) {
        return;
    }
    v += from;
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_cuda_pull_array(v, x, n);
    math21_vector_save_cpu(name, x, 0, n);
    math21_vector_free_cpu(x);
}

// [from, to)
void math21_vector_log_cuda(const char *name, const float *v, size_t from, size_t to) {
    size_t n = to - from;
    v += from;
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_cuda_pull_array(v, x, n);
    math21_vector_log_cpu(name, x, 0, n);
    math21_vector_free_cpu(x);
}

float *math21_vector_create_with_default_value_cuda(size_t n, float value) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    math21_vector_set_cuda(n, value, x_gpu, 1);
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

int *math21_vector_create_with_default_value_int_cuda(size_t n, int value) {
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    math21_vector_set_int_cuda(n, value, x_gpu, 1);
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

void *math21_vector_create_buffer_cuda(size_t n, size_t elementSize) {
    void *x_gpu;
    size_t size = elementSize * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

float *math21_vector_resize_with_default_value_cuda(float *v, size_t n, float value) {
    math21_vector_free_cuda(v);
    v = math21_vector_create_with_default_value_cuda(n, value);
    return v;
}

void *math21_vector_resize_buffer_cuda(void *v, size_t n, size_t elementSize) {
    math21_vector_free_cuda(v);
    v = math21_vector_create_buffer_cuda(n, elementSize);
    return v;
}

int *math21_vector_resize_with_default_value_int_cuda(int *v, size_t n, int value) {
    math21_vector_free_cuda(v);
    v = math21_vector_create_with_default_value_int_cuda(n, value);
    return v;
}

float *math21_vector_create_from_cpuvector_cuda(size_t n, const float *x, int stride_x) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    if (!x) {
        math21_vector_set_cuda(n, 0, x_gpu, 1);
    } else {
        if (stride_x == 1) {
            status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
            math21_cuda_check_error(status);
        } else {
            float *v = math21_vector_create_from_cpuvector_cpu(n, x, stride_x);
            status = cudaMemcpy(x_gpu, v, size, cudaMemcpyHostToDevice);
            math21_cuda_check_error(status);
            math21_vector_free_cpu(v);
        }
    }
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

int *math21_vector_create_from_cpuvector_int_cuda(size_t n, const int *x, int stride_x) {
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaError_t status = cudaMalloc((void **) &x_gpu, size);
    math21_cuda_check_error(status);
    if (!x) {
        math21_vector_set_int_cuda(n, 0, x_gpu, 1);
    } else {
        if (stride_x == 1) {
            status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
            math21_cuda_check_error(status);
        } else {
            int *v = math21_vector_create_from_cpuvector_int_cpu(n, x, stride_x);
            status = cudaMemcpy(x_gpu, v, size, cudaMemcpyHostToDevice);
            math21_cuda_check_error(status);
            math21_vector_free_cpu(v);
        }
    }
    if (!x_gpu) math21_error("Cuda malloc failed\n");
    return x_gpu;
}

void math21_vector_free_cuda(void *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    math21_cuda_check_error(status);
}

__global__ void
math21_vector_mean_fast_cuda_kernel(const float *X, int mini_batch_size, int features_size, int in_class_size,
                                    float *mean) {
    const int threads_size = MATH21_CUDA_BLOCK_SIZE;
    __shared__ float local[threads_size];

    int id = threadIdx.x;
    local[id] = 0;

    int ifeature = blockIdx.x;

    int imember, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; imember += threads_size) {
            int index = imb * features_size * in_class_size + ifeature * in_class_size + imember + id;
            local[id] += (imember + id < in_class_size) ? X[index] : 0;
        }
    }

    __syncthreads();

    // collect results from local
    int i;
    if (id == 0) {
        mean[ifeature] = 0;
        for (i = 0; i < threads_size; ++i) {
            mean[ifeature] += local[i];
        }
        mean[ifeature] /= mini_batch_size * in_class_size;
    }
}

void
math21_vector_mean_fast_cuda(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean) {
    math21_vector_mean_fast_cuda_kernel
            << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                (X, mini_batch_size, features_size, in_class_size, mean);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_mean_cuda_kernel(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean) {
    float scale = 1.f / (mini_batch_size * in_class_size);
    // 3d index (blockIdx.y, blockIdx.x, threadIdx.x) to 1d index
    int ifeature = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (ifeature >= features_size) return;
    int imb, imember;
    mean[ifeature] = 0;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; ++imember) {
            int index = imb * features_size * in_class_size + ifeature * in_class_size + imember;
            mean[ifeature] += X[index];
        }
    }
    mean[ifeature] *= scale;
}

void math21_vector_mean_cuda(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean) {
    math21_vector_mean_cuda_kernel << < math21_cuda_gridsize(features_size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                             (X, mini_batch_size, features_size, in_class_size, mean);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_variance_fast_cuda_kernel(const float *X, const float *mean, int mini_batch_size, int features_size,
                                        int in_class_size, float *variance) {
    const int threads = MATH21_CUDA_BLOCK_SIZE;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int ifeature = blockIdx.x;

    int imember, imb;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; imember += threads) {
            int index = imb * features_size * in_class_size + ifeature * in_class_size + imember + id;

            local[id] += (imember + id < in_class_size) ? powf((X[index] - mean[ifeature]), 2) : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        int i;
        variance[ifeature] = 0;
        for (i = 0; i < threads; ++i) {
            variance[ifeature] += local[i];
        }
        variance[ifeature] /= (in_class_size * mini_batch_size - 1);
    }
}

void math21_vector_variance_fast_cuda(const float *X, const float *mean, int mini_batch_size, int features_size,
                                      int in_class_size, float *variance) {
    math21_vector_variance_fast_cuda_kernel << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                                                (X, mean, mini_batch_size, features_size, in_class_size, variance);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_variance_cuda_kernel(const float *X, const float *mean, int mini_batch_size, int features_size,
                                   int in_class_size, float *variance) {
    float scale = 1.f / (mini_batch_size * in_class_size - 1);
    int j, k;
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= features_size) return;
    variance[i] = 0;
    for (j = 0; j < mini_batch_size; ++j) {
        for (k = 0; k < in_class_size; ++k) {
            int index = j * features_size * in_class_size + i * in_class_size + k;
            variance[i] += powf((X[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

void math21_vector_variance_cuda(const float *X, const float *mean, int mini_batch_size, int features_size,
                                 int in_class_size, float *variance) {
    math21_vector_variance_cuda_kernel << < math21_cuda_gridsize(features_size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                 (X, mean, mini_batch_size, features_size, in_class_size, variance);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_assign_from_vector_N8_cuda_kernel(int n, const NumN8 *X, NumN8 *Y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = X[i];
}

__global__ void
math21_vector_assign_from_vector_with_offset_cuda_kernel(int n, const float *X, int offset_x, int stride_x, float *Y,
                                                         int offset_y,
                                                         int stride_y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) Y[i * stride_y + offset_y] = X[i * stride_x + offset_x];
}

void
math21_vector_assign_from_vector_N8_cuda(int n, const NumN8 *X, NumN8 *Y) {
    math21_vector_assign_from_vector_N8_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                  (n, X, Y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void
math21_vector_assign_from_vector_with_offset_cuda(int n, const float *X, int offset_x, int stride_x, float *Y,
                                                  int offset_y,
                                                  int stride_y) {
    math21_vector_assign_from_vector_with_offset_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                           (n, X, offset_x, stride_x, Y, offset_y, stride_y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void math21_vector_assign_from_vector_cuda(int n, const float *X, int stride_x, float *Y, int stride_y) {
    math21_vector_assign_from_vector_with_offset_cuda(n, X, 0, stride_x, Y, 0, stride_y);
}

void math21_vector_kx_cuda(int n, float k, float *x, int stride_x) {
    math21_generic_vector_kx_wrapper((NumN) n, k, x, (NumN) stride_x, m21_type_NumR32);
}

__global__ void math21_vector_k_add_x_cuda_kernel(int n, float k, float *x, int stride_x) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i * stride_x] += k;
}

void math21_vector_k_add_x_cuda(int n, float k, float *x, int stride_x) {
    math21_vector_k_add_x_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, k, x, stride_x);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_kx_add_y_cuda_kernel(int n, float k, const float *X, int offset_x, int stride_x, float *Y, int offset_y,
                                   int stride_y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) Y[offset_y + i * stride_y] += k * X[offset_x + i * stride_x];
}

void
math21_vector_kx_add_y_with_offset_cuda(int n, float k, const float *X, int offset_x, int stride_x, float *Y,
                                        int offset_y,
                                        int stride_y) {
    math21_vector_kx_add_y_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                     (n, k, X, offset_x, stride_x, Y, offset_y, stride_y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

// y = k*x + y
void math21_vector_kx_add_y_cuda(int n, float k, const float *x, int stride_x, float *y, int stride_y) {
    math21_vector_kx_add_y_with_offset_cuda(n, k, x, 0, stride_x, y, 0, stride_y);
}

__global__ void
math21_vector_normalize_cuda_kernel(int n, float *x, const float *mean, const float *variance, int mini_batch_size,
                                    int features_size, int in_class_size) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int f = (index / in_class_size) % features_size;

    x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
}

void
math21_vector_normalize_cuda(float *x, const float *mean, const float *variance, int mini_batch_size, int features_size,
                             int in_class_size) {
    size_t n = mini_batch_size * features_size * in_class_size;
    math21_vector_normalize_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                      (n, x, mean, variance, mini_batch_size, features_size, in_class_size);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_kx_with_in_class_cuda_kernel(float *x, const float *k, int features_size, int in_class_size) {
    int imember = blockIdx.x * blockDim.x + threadIdx.x;
    int ifeature = blockIdx.y;
    int imb = blockIdx.z;

    if (imember < in_class_size) x[(imb * features_size + ifeature) * in_class_size + imember] *= k[ifeature];
}

void math21_vector_kx_with_in_class_cuda(float *x, const float *k, int mini_batch_size, int features_size,
                                         int in_class_size) {
    dim3
            dimGrid((in_class_size
                     - 1) / MATH21_CUDA_BLOCK_SIZE + 1, features_size, mini_batch_size);
    dim3
            dimBlock(
            MATH21_CUDA_BLOCK_SIZE, 1, 1);

    math21_vector_kx_with_in_class_cuda_kernel << < dimGrid, dimBlock >> > (x, k, features_size, in_class_size);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_x_add_b_with_in_class_cuda_kernel(float *x, const float *b, int mini_batch_size, int features_size,
                                                int in_class_size) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= mini_batch_size * features_size * in_class_size) return;
    int imember = index % in_class_size;
    index /= in_class_size;
    int ifeature = index % features_size;
    index /= features_size;
    int imb = index;

    x[(imb * features_size + ifeature) * in_class_size + imember] += b[ifeature];
}

void math21_vector_x_add_b_with_in_class_cuda(float *x, const float *b, int mini_batch_size, int features_size,
                                              int in_class_size) {
    int num = mini_batch_size * features_size * in_class_size;
    math21_vector_x_add_b_with_in_class_cuda_kernel << < math21_cuda_gridsize(num), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                    (x, b, mini_batch_size, features_size, in_class_size);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_sum_with_in_class_conn_cuda_kernel(float *db, const float *dY, int mini_batch_size, int features_size) {
    int ifeature = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (ifeature >= features_size) return;
    int imb;
    float sum = 0;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        int i = imb * features_size + ifeature;
        sum += dY[i];
    }
    db[ifeature] += sum;
}

__global__ void
math21_vector_sum_with_in_class_cuda_kernel(float *db, const float *dY, int mini_batch_size, int features_size,
                                            int in_class_size) {
    __shared__ float part[MATH21_CUDA_BLOCK_SIZE];
    int imember, imb;
    int ifeature = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; imember += MATH21_CUDA_BLOCK_SIZE) {
            int index = p + (imb * features_size + ifeature) * in_class_size + imember;
            sum += (p + imember < in_class_size) ? dY[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        int i;
        for (i = 0; i < MATH21_CUDA_BLOCK_SIZE; ++i) db[ifeature] += part[i];
    }
}

void math21_vector_sum_with_in_class_cuda(float *db, const float *dY, int mini_batch_size, int features_size,
                                          int in_class_size) {
    if (in_class_size == 1) {
        math21_vector_sum_with_in_class_conn_cuda_kernel << < math21_cuda_gridsize(features_size),
                MATH21_CUDA_BLOCK_SIZE >> > (db, dY, mini_batch_size, features_size);
    } else {
        math21_vector_sum_with_in_class_cuda_kernel << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                                                        (db, dY, mini_batch_size, features_size, in_class_size);
    }
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_sum_SchurProduct_with_in_class_cuda_kernel(const float *X, const float *dY, int mini_batch_size,
                                                         int features_size,
                                                         int in_class_size, float *dk) {
    __shared__ float part[MATH21_CUDA_BLOCK_SIZE];
    int imb, imember;
    int ifeature = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for (imb = 0; imb < mini_batch_size; ++imb) {
        for (imember = 0; imember < in_class_size; imember += MATH21_CUDA_BLOCK_SIZE) {
            int index = p + (imb * features_size + ifeature) * in_class_size + imember;
            sum += (p + imember < in_class_size) ? dY[index] * X[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        int i;
        for (i = 0; i < MATH21_CUDA_BLOCK_SIZE; ++i) dk[ifeature] += part[i];
    }
}

void math21_vector_sum_SchurProduct_with_in_class_cuda(const float *X, const float *dY, int mini_batch_size,
                                                       int features_size,
                                                       int in_class_size, float *dk) {
    math21_vector_sum_SchurProduct_with_in_class_cuda_kernel << < features_size, MATH21_CUDA_BLOCK_SIZE >> >
                                                                                 (X, dY, mini_batch_size, features_size, in_class_size, dk);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_set_cuda_kernel(int n, float value, float *X, int stride_x) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) X[i * stride_x] = value;
}

void math21_vector_set_cuda(int n, float value, float *X, int stride_x) {
    math21_vector_set_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, value, X, stride_x);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_set_int_kernel(int n, int value, int *X, int stride_x) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) X[i * stride_x] = value;
}

void math21_vector_set_int_cuda(int n, int value, int *X, int stride_x) {
    math21_vector_set_int_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, value, X, stride_x);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_assign_3d_d2_cuda_kernel(int size, const float *data1, float *data2,
                                       int d1, int d2, int d3, int d2y, int offset2, int isToSmall) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i1, i2_y, ix, iy, i2, i3;
    iy = id;
    i3 = iy % d3;
    iy = iy / d3;
    i2_y = iy % d2y;
    i1 = iy / d2y;

    i2 = i2_y + offset2;
    ix = i1 * d2 * d3 + i2 * d3 + i3;
    iy = id;
    if (isToSmall) {
        data2[iy] = data1[ix];
    } else {
        data2[ix] = data1[iy];
    }
}

// x -> y, (d1, d2, d3) -> (d1, d2y, d3) with d2 >= d2y
// x <- y, (d1, d2, d3) <- (d1, d2y, d3) with d2 >= d2y
void math21_vector_assign_3d_d2_cuda(const float *data1, float *data2,
                                     int d1, int d2, int d3, int d2y, int offset2, int isToSmall) {
    size_t size = d1 * d2y * d3;
    math21_vector_assign_3d_d2_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                            (size, data1, data2, d1, d2, d3, d2y, offset2, isToSmall);
}

__global__ void
math21_vector_transpose_d1234_to_d1324_cuda_kernel(int size, const float *x, float *y, int d1, int d2, int d3, int d4) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i1, i2_y, i3_y, i4, ix, iy, d3_y, d2_y, i2, i3;
    iy = id;
    d3_y = d2;
    d2_y = d3;
    i4 = iy % d4;
    iy = iy / d4;
    i3_y = iy % d3_y;
    iy = iy / d3_y;
    i2_y = iy % d2_y;
    i1 = iy / d2_y;

    i2 = i3_y;
    i3 = i2_y;
    ix = i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4;
    iy = id;
    y[iy] = x[ix];
}

// (d1, d2, d3, d4) -> (d1, d3, d2, d4)
void math21_vector_transpose_d1234_to_d1324_cuda(const float *x, float *y, int d1, int d2, int d3, int d4) {
    size_t size = d1 * d2 * d3 * d4;
    math21_vector_transpose_d1234_to_d1324_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                        (size, x, y, d1, d2, d3, d4);
}

__global__ void math21_vector_feature2d_add_2_cuda_kernel(int size, int mini_batch_size,
                                                          int nch, int nr, int nc,
                                                          float kx, const float *X, int nch_X, int nr_X, int nc_X,
                                                          float stride_r_x, float stride_c_x,
                                                          float ky, float *Y, int nch_Y, int nr_Y, int nc_Y,
                                                          float stride_r_y, float stride_c_y) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int ic = id % nc;
    id /= nc;
    int ir = id % nr;
    id /= nr;
    int ich = id % nch;
    id /= nch;
    int imb = id % mini_batch_size;

    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);
    Y[index_Y] = kx * X[index_X] + ky * Y[index_Y];
}

__global__ void math21_vector_feature2d_add_3_cuda_kernel(int size, int mini_batch_size,
                                                          int nch, int nr, int nc,
                                                          float kx, const float *X, int nch_X, int nr_X, int nc_X,
                                                          float stride_r_x, float stride_c_x,
                                                          float kx2, const float *X2, int nch_X2, int nr_X2, int nc_X2,
                                                          float stride_r_x2, float stride_c_x2,
                                                          float ky, float *Y, int nch_Y, int nr_Y, int nc_Y,
                                                          float stride_r_y, float stride_c_y) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int ic = id % nc;
    id /= nc;
    int ir = id % nr;
    id /= nr;
    int ich = id % nch;
    id /= nch;
    int imb = id % mini_batch_size;

    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);
    // X2(imb, ich, ir*stride_r_x2, ic*stride_c_x2)
    int index_X2 = ((imb * nch_X2 + ich) * nr_X2 + (int) (ir * stride_r_x2)) * nc_X2 + (int) (ic * stride_c_x2);
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);
    Y[index_Y] = kx * X[index_X] + kx2 * X2[index_X2] + ky * Y[index_Y];
}

void math21_vector_feature2d_add_2_cuda(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y) {
    int nch = math21_number_min_2_int(nch_X, nch_Y);
    int nr = math21_number_min_2_int(nr_X, nr_Y);
    int nc = math21_number_min_2_int(nc_X, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_y = (float) nc_Y / nc;

    int size = mini_batch_size * nch * nr * nc;
    math21_vector_feature2d_add_2_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                               (size, mini_batch_size,
                                                                                       nch, nr, nc,
                                                                                       kx, X, nch_X, nr_X, nc_X,
                                                                                       stride_r_x, stride_c_x,
                                                                                       ky, Y, nch_Y, nr_Y, nc_Y,
                                                                                       stride_r_y, stride_c_y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void math21_vector_feature2d_add_3_cuda(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float kx2, const float *X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y) {

    int nch = math21_number_min_3_int(nch_X, nch_X2, nch_Y);
    int nr = math21_number_min_3_int(nr_X, nr_X2, nr_Y);
    int nc = math21_number_min_3_int(nc_X, nc_X2, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_x2 = (float) nr_X2 / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_x2 = (float) nc_X2 / nc;
    float stride_c_y = (float) nc_Y / nc;

    int size = mini_batch_size * nch * nr * nc;
    math21_vector_feature2d_add_3_cuda_kernel << < math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                                                               (size, mini_batch_size,
                                                                                       nch, nr, nc,
                                                                                       kx, X, nch_X, nr_X, nc_X,
                                                                                       stride_r_x, stride_c_x,
                                                                                       kx2, X2, nch_X2, nr_X2, nc_X2,
                                                                                       stride_r_x2, stride_c_x2,
                                                                                       ky, Y, nch_Y, nr_Y, nc_Y,
                                                                                       stride_r_y, stride_c_y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

// X shape <= Y shape
__global__ void math21_vector_feature2d_sumdownsample_cuda_kernel(size_t n, int mini_batch_size,
                                                                  float *X, int nch_X, int nr_X, int nc_X,
                                                                  int stride_X, float k, const float *Y) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int index_X = i;
    int ic_X = i % nc_X;
    i = i / nc_X;
    int ir_X = i % nr_X;
    i = i / nr_X;
    int ich_X = i % nch_X;
    i = i / nch_X;
    int imb_X = i % mini_batch_size;

    int ic_Y_abs = ic_X * stride_X;
    int ir_Y_abs = ir_X * stride_X;
    int ich_Y = ich_X;
    int imb_Y = imb_X;

    int nc_Y = nc_X * stride_X;
    int nr_Y = nr_X * stride_X;
    int nch_Y = nch_X;

    int ksize = stride_X;
    for (int ir_K = 0; ir_K < ksize; ++ir_K) {
        for (int ic_K = 0; ic_K < ksize; ++ic_K) {
            int ir_Y = ir_Y_abs + ir_K;
            int ic_Y = ic_Y_abs + ic_K;
            int index_Y = ((imb_Y * nch_Y + ich_Y) * nr_Y + ir_Y) * nc_Y + ic_Y;
            X[index_X] += k * Y[index_Y];
        }
    }
}

void
math21_vector_feature2d_sumdownsample_cuda(int mini_batch_size, float *X, int nch_X, int nr_X, int nc_X, int stride_X,
                                           float k, const float *Y) {
    size_t n = mini_batch_size * nch_X * nr_X * nc_X;
    math21_vector_feature2d_sumdownsample_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                    (n, mini_batch_size,
                                                                                            X, nch_X, nr_X, nc_X,
                                                                                            stride_X, k, Y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

// Note: when eps = 0.000000001, atomicAdd will lead to error.
// So use single thread when necessary.
__global__ void math21_vector_feature2d_upsample_cuda_kernel(size_t n, int mini_batch_size,
                                                             float *X, int nch_X, int nr_X, int nc_X,
                                                             int stride_X, int is_upsample, float k, float *Y) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int index_Y = i;
    int ic_Y = i % (nc_X * stride_X);
    i = i / (nc_X * stride_X);
    int ir_Y = i % (nr_X * stride_X);
    i = i / (nr_X * stride_X);
    int ich_Y = i % nch_X;
    i = i / nch_X;
    int imb_Y = i % mini_batch_size;

    int ic_X = ic_Y / stride_X;
    int ir_X = ir_Y / stride_X;
    int ich_X = ich_Y;

    int index_X = imb_Y * nch_X * nr_X * nc_X + ich_X * nr_X * nc_X + ir_X * nc_X + ic_X;

    if (is_upsample) {
        Y[index_Y] += k * X[index_X];
    } else {
        // deprecate, use math21_vector_feature2d_sumdownsample_cuda
        atomicAdd(X + index_X, k * Y[index_Y]);
    }
}

// X shape <= Y shape
void math21_vector_feature2d_upsample_cuda(int mini_batch_size, float *X, int nch_X, int nr_X, int nc_X, int stride_X,
                                           int is_upsample, float k, float *Y) {
    size_t n = mini_batch_size * nch_X * nr_X * nc_X * stride_X * stride_X;
    math21_vector_feature2d_upsample_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                               (n, mini_batch_size,
                                                                                       X, nch_X, nr_X, nc_X,
                                                                                       stride_X, is_upsample, k, Y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

// X shape <= Y shape
void math21_vector_feature2d_sample_cuda(int mini_batch_size, float *X, int nch_X, int nr_X, int nc_X, int stride_X,
                                         int is_upsample, float k, float *Y) {
    if (is_upsample) {
        math21_vector_feature2d_upsample_cuda(mini_batch_size, X, nch_X, nr_X, nc_X, stride_X,
                                              is_upsample, k, Y);
    } else {
        math21_vector_feature2d_sumdownsample_cuda(mini_batch_size, X, nch_X, nr_X, nc_X, stride_X,
                                                   k, Y);
    }
}

__global__ void math21_vector_clip_cuda_kernel(int n, float k, float *x, int stride_x) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i * stride_x] = fminf(k, fmaxf(-k, x[i * stride_x]));
}

// clip x, so -k <= x <= k
void math21_vector_clip_cuda(int n, float k, float *x, int stride_x) {
    math21_vector_clip_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, k, x, stride_x);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_xy_cuda_kernel(int n, const float *x, int stride_x, float *y, int stride_y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i * stride_y] *= x[i * stride_x];
}

void math21_vector_xy_cuda(int n, const float *x, int stride_x, float *y, int stride_y) {
    math21_vector_xy_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                               (n, x, stride_x, y, stride_y);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void
math21_vector_assign_by_mask_cuda_kernel(int n, float *x, float mask_num, const float *mask, float val) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n && mask[i] == mask_num) x[i] = val;
}

void math21_vector_assign_by_mask_cuda(int n, float *x, float mask_num, const float *mask, float val) {
    math21_vector_assign_by_mask_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                           (n, x, mask_num, mask, val);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_kx_by_mask_cuda_kernel(int n, float k, float *x, const float *mask, float mask_num) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n && mask[i] == mask_num) x[i] *= k;
}

void math21_vector_kx_by_mask_cuda(int n, float k, float *x, const float *mask, float mask_num) {
    math21_vector_kx_by_mask_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                       (n, k, x, mask, mask_num);
    math21_cuda_check_error(cudaPeekAtLastError());
}


void math21_vector_pr_rand_uniform_01_cuda(float *x_gpu, size_t n) {
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = math21_cuda_get_device();
    if (!init[i]) {
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], math21_c_seed_get());
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_loss_l1_cuda_kernel(int n, const float *x, const float *t, float *dx, float *error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = t[i] - x[i];
        error[i] = abs(diff);
        dx[i] = diff > 0 ? 1 : -1;
    }
}

void math21_vector_loss_l1_cuda(int n, const float *x, const float *t, float *dx, float *error) {
    math21_vector_loss_l1_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, x, t, dx, error);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_loss_l2_cuda_kernel(int n, const float *x, const float *t, float *dx, float *error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = t[i] - x[i];
        error[i] = diff * diff;
        dx[i] = diff;
    }
}

void math21_vector_loss_l2_cuda(int n, const float *x, const float *t, float *dx, float *error) {
    math21_vector_loss_l2_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (n, x, t, dx, error);
    math21_cuda_check_error(cudaPeekAtLastError());
}


__global__ void
math21_vector_loss_smooth_l1_cuda_kernel(int n, const float *x, const float *t, float *dx, float *error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = t[i] - x[i];
        float abs_val = fabsf(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            dx[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            dx[i] = (diff > 0) ? 1 : -1;
        }
    }
}

void math21_vector_loss_smooth_l1_cuda(int n, const float *x, const float *t, float *dx, float *error) {
    math21_vector_loss_smooth_l1_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                           (n, x, t, dx, error);
    math21_cuda_check_error(cudaPeekAtLastError());
}

__global__ void math21_vector_zero_by_thresh_cuda_kernel(int n, float *x, int stride_x, float thresh) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        if (fabsf(x[i * stride_x]) < thresh) x[i * stride_x] = 0;
    }
}

// x(i) = 0  if |x(i)| < thresh, for all i
void math21_vector_zero_by_thresh_cuda(int n, float *x, int stride_x, float thresh) {
    math21_vector_zero_by_thresh_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                           (n, x, stride_x, thresh);
    math21_cuda_check_error(cudaPeekAtLastError());
}