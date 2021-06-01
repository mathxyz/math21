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
#include "inner_c.h"
#include "inner_cc.h"
#include "../../generic/files_c.h"
#include "vector_cpu.h"
#include "vector_cuda.h"
#include "vector_opencl.h"
#include "vector_wrapper.h"

using namespace math21;

PointerFloatWrapper math21_vector_deserialize_c_wrapper(FILE *f, size_t *n) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_deserialize_c_cpu(f, n);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_deserialize_c_cuda(f, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_deserialize_c_opencl(f, n);
#endif
}

PointerFloatWrapper math21_vector_deserialize_from_file_wrapper(const char *name, size_t *n) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_deserialize_from_file_cpu(name, n);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_deserialize_from_file_cuda(name, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_deserialize_from_file_opencl(name, n);
#endif
}

void math21_vector_serialize_c_wrapper(FILE *f, PointerFloatInputWrapper v, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_serialize_c_cpu(f, v, n);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_serialize_c_cuda(f, v, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_serialize_c_opencl(f, v, n);
#endif
}

void math21_vector_serialize_to_file_wrapper(const char *name, PointerFloatInputWrapper v, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_serialize_to_file_cpu(name, v, n);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_serialize_to_file_cuda(name, v, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_serialize_to_file_opencl(name, v, n);
#endif
}

void math21_vector_save_wrapper(const char *name, PointerFloatInputWrapper v, size_t from, size_t to) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_save_cpu(name, v, from, to);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_save_cuda(name, v, from, to);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_save_opencl(name, v, from, to);
#endif
}

void math21_vector_log_wrapper(const char *name, PointerFloatInputWrapper v, size_t from, size_t to) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_log_cpu(name, v, from, to);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_log_cuda(name, v, from, to);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_log_opencl(name, v, from, to);
#endif
}

PointerFloatWrapper math21_vector_create_with_default_value_wrapper(size_t n, float value) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_create_with_default_value_cpu(n, value);

#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_create_with_default_value_cuda(n, value);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_create_with_default_value_opencl(n, value);
#endif
}

PointerVoidWrapper math21_vector_create_buffer_wrapper(size_t n, size_t elementSize) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_create_buffer_cpu(n, elementSize);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_create_buffer_cuda(n, elementSize);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_create_buffer_opencl(n, elementSize);
#endif
}

PointerFloatWrapper math21_vector_resize_with_default_value_wrapper(PointerFloatWrapper v, size_t n, float value) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_resize_with_default_value_cpu(v, n, value);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_resize_with_default_value_cuda(v, n, value);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_resize_with_default_value_opencl(v, n, value);
#endif
}

PointerVoidWrapper math21_vector_resize_buffer_wrapper(PointerVoidWrapper v, size_t n, size_t elementSize) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_resize_buffer_cpu(v, n, elementSize);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_resize_buffer_cuda(v, n, elementSize);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_resize_buffer_opencl(v, n, elementSize);
#endif
}

PointerIntWrapper math21_vector_resize_with_default_value_int_wrapper(PointerIntWrapper v, size_t n, int value) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_resize_with_default_value_int_cpu(v, n, value);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_resize_with_default_value_int_cuda(v, n, value);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_resize_with_default_value_int_opencl(v, n, value);
#endif
}

PointerFloatWrapper math21_vector_create_from_cpuvector_wrapper(size_t n, const float *x, int stride_x) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_create_from_cpuvector_cpu(n, x, stride_x);

#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_create_from_cpuvector_cuda(n, x, stride_x);

#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_create_from_cpuvector_opencl(n, x, stride_x);
#endif
}

PointerIntWrapper math21_vector_create_from_cpuvector_int_wrapper(size_t n, const int *x, int stride_x) {
#if defined(MATH21_FLAG_USE_CPU)
    return math21_vector_create_from_cpuvector_int_cpu(n, x, stride_x);
#elif defined(MATH21_FLAG_USE_CUDA)
    return math21_vector_create_from_cpuvector_int_cuda(n, x, stride_x);
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_vector_create_from_cpuvector_int_opencl(n, x, stride_x);
#endif
}

void math21_vector_free_wrapper(PointerVoidWrapper x) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_free_cpu(x);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_free_cuda(x);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_free_opencl(x);
#endif
}

// mu = E(X)
// X shape: mini_batch_size*features_size*in_class_size
// rnn: in_class_size=1
// cnn: in_class_size=nr_Y*nc_Y
void math21_vector_mean_wrapper(PointerFloatInputWrapper X, int mini_batch_size, int features_size, int in_class_size,
                                PointerFloatWrapper mean) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_mean_cpu(X, mini_batch_size, features_size, in_class_size, mean);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_mean_fast_cuda(X, mini_batch_size, features_size, in_class_size, mean);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_mean_fast_opencl(X, mini_batch_size, features_size, in_class_size, mean);
#endif
}

void
math21_vector_variance_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper mean, int mini_batch_size,
                               int features_size,
                               int in_class_size,
                               PointerFloatWrapper variance) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_variance_cpu(X, mean, mini_batch_size, features_size, in_class_size, variance);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_variance_fast_cuda(X, mean, mini_batch_size, features_size, in_class_size, variance);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_variance_fast_opencl(X, mean, mini_batch_size, features_size, in_class_size, variance);
#endif
}

void math21_vector_assign_from_vector_N8_wrapper(int n, PointerN8InputWrapper x, PointerN8Wrapper y) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_assign_from_vector_N8_cpu(n, x, y);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_assign_from_vector_N8_cuda(n, x, y);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_assign_from_vector_N8_opencl(n, x, y);
#endif
}

void math21_vector_assign_from_vector_wrapper(int n, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y,
                                              int stride_y) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_assign_from_vector_cpu(n, x, stride_x, y, stride_y);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_assign_from_vector_cuda(n, x, stride_x, y, stride_y);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_assign_from_vector_opencl(n, x, stride_x, y, stride_y);
#endif
}

// deprecate, use math21_generic_vector_kx_wrapper instead.
void math21_vector_kx_wrapper(int n, float k, PointerFloatWrapper x, int stride_x) {
    math21_generic_vector_kx_wrapper(n, k, x, stride_x, m21_type_NumR32);
//#if defined(MATH21_FLAG_USE_CPU)
//    math21_vector_kx_cpu(n, k, x, stride_x);
//
//#elif defined(MATH21_FLAG_USE_CUDA)
//    math21_vector_kx_cuda(n, k, x, stride_x);
//
//#elif defined(MATH21_FLAG_USE_OPENCL)
//    math21_vector_kx_opencl(n, k, x, stride_x);
//#endif
}

void math21_vector_k_add_x_wrapper(int n, float k, PointerFloatWrapper x, int stride_x) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_k_add_x_cpu(n, k, x, stride_x);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_k_add_x_cuda(n, k, x, stride_x);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_k_add_x_opencl(n, k, x, stride_x);
#endif
}

// deprecate, use math21_generic_vector_kx_add_y_wrapper.
void math21_vector_kx_add_y_wrapper(int n, float k, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y,
                                    int stride_y) {
    math21_generic_vector_kx_add_y_wrapper(n, k, x, stride_x, y, stride_y, m21_type_NumR32);
//#if defined(MATH21_FLAG_USE_CPU)
//    math21_vector_kx_add_y_cpu(n, k, x, stride_x, y, stride_y);
//
//#elif defined(MATH21_FLAG_USE_CUDA)
//    math21_vector_kx_add_y_cuda(n, k, x, stride_x, y, stride_y);
//#elif defined(MATH21_FLAG_USE_OPENCL)
//    math21_vector_kx_add_y_opencl(n, k, x, stride_x, y, stride_y);
//#endif
}

void
math21_vector_normalize_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper mean, PointerFloatInputWrapper variance,
                                int mini_batch_size, int features_size,
                                int in_class_size) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_normalize_cpu(x, mean, variance, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_normalize_cuda(x, mean, variance, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_normalize_opencl(x, mean, variance, mini_batch_size, features_size, in_class_size);
#endif
}

void
math21_vector_kx_with_in_class_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper k, int mini_batch_size,
                                       int features_size,
                                       int in_class_size) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_kx_with_in_class_cpu(x, k, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_kx_with_in_class_cuda(x, k, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_kx_with_in_class_opencl(x, k, mini_batch_size, features_size, in_class_size);
#endif
}

void math21_vector_x_add_b_with_in_class_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper b, int mini_batch_size,
                                                 int features_size,
                                                 int in_class_size) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_x_add_b_with_in_class_cpu(x, b, mini_batch_size, features_size, in_class_size);

#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_x_add_b_with_in_class_cuda(x, b, mini_batch_size, features_size, in_class_size);

#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_x_add_b_with_in_class_opencl(x, b, mini_batch_size, features_size, in_class_size);
#endif
}

float math21_vector_sum(const float *x, int n) {
    return math21_vector_sum_cpu(x, n);
}

void math21_vector_sum_with_in_class_wrapper(PointerFloatWrapper db, PointerFloatInputWrapper dY, int mini_batch_size,
                                             int features_size,
                                             int in_class_size) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_sum_with_in_class_cpu(db, dY, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_sum_with_in_class_cuda(db, dY, mini_batch_size, features_size, in_class_size);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_sum_with_in_class_opencl(db, dY, mini_batch_size, features_size, in_class_size);
#endif
}

void math21_vector_sum_SchurProduct_with_in_class_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper dY,
                                                          int mini_batch_size, int features_size,
                                                          int in_class_size, PointerFloatWrapper dk) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_sum_SchurProduct_with_in_class_cpu(X, dY, mini_batch_size, features_size, in_class_size, dk);

#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_sum_SchurProduct_with_in_class_cuda(X, dY, mini_batch_size, features_size, in_class_size, dk);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_sum_SchurProduct_with_in_class_opencl(X, dY, mini_batch_size, features_size, in_class_size, dk);
#endif
}

// use math21_generic_vector_set_by_value_wrapper
void math21_vector_set_wrapper(int n, float value, PointerFloatWrapper X, int stride_x) {
    math21_generic_vector_set_by_value_wrapper(n, value, X, stride_x, m21_type_NumR32);
//#if defined(MATH21_FLAG_USE_CPU)
//    math21_vector_set_cpu(n, value, X, stride_x);
//
//#elif defined(MATH21_FLAG_USE_CUDA)
//    math21_vector_set_cuda(n, value, X, stride_x);
//#elif defined(MATH21_FLAG_USE_OPENCL)
//    math21_vector_set_opencl(n, value, X, stride_x);
//#endif
}

void math21_vector_set_int_wrapper(int n, int value, PointerIntWrapper X, int stride_x) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_set_int_cpu(n, value, X, stride_x);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_set_int_cuda(n, value, X, stride_x);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_set_int_opencl(n, value, X, stride_x);
#endif
}

void math21_vector_assign_3d_d2_wrapper(PointerFloatInputWrapper data1, PointerFloatWrapper data2,
                                        int d1, int d2, int d3, int d2y, int offset2, int isToSmall) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_assign_3d_d2_cpu(data1, data2, d1, d2, d3, d2y, offset2, isToSmall);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_assign_3d_d2_cuda(data1, data2, d1, d2, d3, d2y, offset2, isToSmall);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_tool_assert(0);
#endif
}

void math21_vector_transpose_d1234_to_d1324_wrapper(PointerFloatInputWrapper x, PointerFloatWrapper y,
                                                    int d1, int d2, int d3, int d4) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_transpose_d1234_to_d1324_cpu(x, y, d1, d2, d3, d4);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_transpose_d1234_to_d1324_cuda(x, y, d1, d2, d3, d4);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_tool_assert(0);
#endif
}

void math21_vector_feature2d_add_2_wrapper(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_feature2d_add_2_cpu(mini_batch_size,
                                      kx, X, nch_X, nr_X, nc_X,
                                      ky, Y, nch_Y, nr_Y, nc_Y);

#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_feature2d_add_2_cuda(mini_batch_size,
                                       kx, X, nch_X, nr_X, nc_X,
                                       ky, Y, nch_Y, nr_Y, nc_Y);

#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_feature2d_add_2_opencl(mini_batch_size,
                                         kx, X, nch_X, nr_X, nc_X,
                                         ky, Y, nch_Y, nr_Y, nc_Y);

#endif
}

void math21_vector_feature2d_add_3_wrapper(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float kx2, PointerFloatInputWrapper X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_feature2d_add_3_cpu(mini_batch_size,
                                      kx, X, nch_X, nr_X, nc_X,
                                      kx2, X2, nch_X2, nr_X2, nc_X2,
                                      ky, Y, nch_Y, nr_Y, nc_Y);

#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_feature2d_add_3_cuda(mini_batch_size,
                                       kx, X, nch_X, nr_X, nc_X,
                                       kx2, X2, nch_X2, nr_X2, nc_X2,
                                       ky, Y, nch_Y, nr_Y, nc_Y);

#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_feature2d_add_3_opencl(mini_batch_size,
                                         kx, X, nch_X, nr_X, nc_X,
                                         kx2, X2, nch_X2, nr_X2, nc_X2,
                                         ky, Y, nch_Y, nr_Y, nc_Y);

#endif
}

// todo: check cpu, checked, now need testing
void math21_vector_feature2d_sample_wrapper(
        int mini_batch_size,
        PointerFloatWrapper X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k,
        PointerFloatWrapper Y) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_feature2d_sample_cpu(
            mini_batch_size, X, nch_X, nr_X, nc_X, stride_X, is_upsample, k, Y);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_feature2d_sample_cuda(
            mini_batch_size, X, nch_X, nr_X, nc_X, stride_X, is_upsample, k, Y);

#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_feature2d_sample_opencl(
            mini_batch_size, X, nch_X, nr_X, nc_X, stride_X, is_upsample, k, Y);
#endif
}

void math21_vector_clip_wrapper(int n, float k, PointerFloatWrapper x, int stride) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_clip_cpu(n, k, x, stride);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_clip_cuda(n, k, x, stride);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_clip_opencl(n, k, x, stride);
#endif
}

// use math21_generic_vector_xy_wrapper. Note index from 1 or 1
void math21_vector_xy_wrapper(int n, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y, int stride_y) {
    math21_generic_vector_xy_wrapper(n, x, stride_x, y, stride_y, m21_type_NumR32);
//#if defined(MATH21_FLAG_USE_CPU)
//    math21_vector_xy_cpu(n, x, stride_x, y, stride_y);
//#elif defined(MATH21_FLAG_USE_CUDA)
//    math21_vector_xy_cuda(n, x, stride_x, y, stride_y);
//#elif defined(MATH21_FLAG_USE_OPENCL)
//    math21_vector_xy_opencl(n, x, stride_x, y, stride_y);
//#endif
}

// may deprecate
void math21_vector_assign_by_mask_wrapper(int n, PointerFloatWrapper x, float mask_num, PointerFloatInputWrapper mask,
                                          float val) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_tool_assert(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_assign_by_mask_cuda(n, x, mask_num, mask, val);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_assign_by_mask_opencl(n, x, mask_num, mask, val);
#endif
}

void math21_vector_kx_by_mask_wrapper(int n, float k, PointerFloatWrapper x, PointerFloatInputWrapper mask,
                                      float mask_num) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_tool_assert(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_kx_by_mask_cuda(n, k, x, mask, mask_num);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_kx_by_mask_opencl(n, k, x, mask, mask_num);
#endif
}

NumB math21_vector_isEmpty_wrapper(PointerVoidInputWrapper x) {
#if defined(MATH21_FLAG_USE_CPU)
    return !x ? 1 : 0;
#elif defined(MATH21_FLAG_USE_CUDA)
    return !x ? 1 : 0;
#elif defined(MATH21_FLAG_USE_OPENCL)
    return math21_opencl_vector_isEmpty(&x);
#endif
}

PointerVoidWrapper math21_vector_getEmpty_wrapper() {
#if defined(MATH21_FLAG_USE_CPU)
    return 0;
#elif defined(MATH21_FLAG_USE_CUDA)
    return 0;
#elif defined(MATH21_FLAG_USE_OPENCL)
    PointerVoidWrapper x;
    math21_opencl_vector_reset(&x);
    return x;
#endif
}

PointerN8Wrapper math21_vector_getEmpty_N8_wrapper() {
    return (PointerN8Wrapper) math21_vector_getEmpty_wrapper();
}

PointerFloatWrapper math21_vector_getEmpty_R32_wrapper() {
    return (PointerFloatWrapper) math21_vector_getEmpty_wrapper();
}

void math21_vector_push_wrapper(PointerFloatWrapper x_gpu, const float *x, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_push_array(x_gpu, x, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_push_array(x_gpu, x, n);
#endif
}

void math21_vector_push_N8_wrapper(PointerN8Wrapper x_gpu, const NumN8 *x, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_push_N8_array(x_gpu, x, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_push_N8_array(x_gpu, x, n);
#endif
}

void math21_vector_pull_wrapper(PointerFloatInputWrapper x_gpu, float *x, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_pull_array(x_gpu, x, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_pull_array(x_gpu, x, n);
#endif
}

void math21_vector_pull_N8_wrapper(PointerN8InputWrapper x_gpu, NumN8 *x, size_t n) {
#if defined(MATH21_FLAG_USE_CPU)
    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_pull_N8_array(x_gpu, x, n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_pull_N8_array(x_gpu, x, n);
#endif
}

void math21_vector_log_pointer_wrapper(PointerFloatInputWrapper v) {
#if defined(MATH21_FLAG_USE_CPU)
    printf("cpu pointer: %p\n", v);
#elif defined(MATH21_FLAG_USE_CUDA)
    printf("cuda pointer: %p\n", v);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_vector_log_pointer(v);
#endif
}

void math21_vector_pr_rand_uniform_01_wrapper(PointerFloatWrapper v, int size) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_pr_rand_uniform_01_cpu(v, size);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_pr_rand_uniform_01_cuda(v, size);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_pr_rand_uniform_01_opencl(v, size);
#endif
}

void math21_vector_loss_l1_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                   PointerFloatWrapper dx, PointerFloatWrapper error) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_loss_l1_cpu(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_loss_l1_cuda(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_loss_l1_opencl(n, x, t, dx, error);
#endif
}

void math21_vector_loss_l2_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                   PointerFloatWrapper dx, PointerFloatWrapper error) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_loss_l2_cpu(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_loss_l2_cuda(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_loss_l2_opencl(n, x, t, dx, error);
#endif
}

void math21_vector_loss_smooth_l1_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                          PointerFloatWrapper dx, PointerFloatWrapper error) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_vector_loss_smooth_l1_cpu(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_loss_smooth_l1_cuda(n, x, t, dx, error);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_loss_smooth_l1_opencl(n, x, t, dx, error);
#endif
}

void math21_vector_zero_by_thresh_wrapper(int n, PointerFloatWrapper x, int stride_x, float thresh) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_tool_assert(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_vector_zero_by_thresh_cuda(n, x, stride_x, thresh);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_vector_zero_by_thresh_opencl(n, x, stride_x, thresh);
#endif
}
