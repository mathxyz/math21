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

PointerFloatWrapper math21_vector_deserialize_c_wrapper(FILE *f, size_t *n);

PointerFloatWrapper math21_vector_deserialize_from_file_wrapper(const char *name, size_t *n);

void math21_vector_serialize_c_wrapper(FILE *f, PointerFloatInputWrapper v, size_t n);

void math21_vector_serialize_to_file_wrapper(const char *name, PointerFloatInputWrapper v, size_t n);

void math21_vector_save_wrapper(const char *name, PointerFloatInputWrapper v, size_t from, size_t to);

void math21_vector_log_wrapper(const char *name, PointerFloatInputWrapper v, size_t from, size_t to);

PointerFloatWrapper math21_vector_create_with_default_value_wrapper(size_t n, float value);

PointerVoidWrapper math21_vector_create_buffer_wrapper(size_t n, size_t elementSize);

// data not kept
PointerFloatWrapper math21_vector_resize_with_default_value_wrapper(PointerFloatWrapper v, size_t n, float value);

// data not kept
PointerVoidWrapper math21_vector_resize_buffer_wrapper(PointerVoidWrapper v, size_t n, size_t elementSize);

// data not kept
PointerIntWrapper math21_vector_resize_with_default_value_int_wrapper(PointerIntWrapper v, size_t n, int value);

PointerFloatWrapper math21_vector_create_from_cpuvector_wrapper(size_t n, const float *x, int stride_x);

PointerIntWrapper math21_vector_create_from_cpuvector_int_wrapper(size_t n, const int *x, int stride_x);

void math21_vector_free_wrapper(PointerVoidWrapper x);

void
math21_vector_mean_wrapper(PointerFloatInputWrapper x, int batch, int filters, int spatial, PointerFloatWrapper mean);

void
math21_vector_variance_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper mean, int mini_batch_size,
                               int features_size,
                               int in_class_size,
                               PointerFloatWrapper variance);

void math21_vector_assign_from_vector_N8_wrapper(int n, PointerN8InputWrapper x, PointerN8Wrapper y);

void math21_vector_assign_from_vector_wrapper(int n, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y,
                                              int stride_y);

void math21_vector_kx_wrapper(int n, float k, PointerFloatWrapper x, int stride_x);

void math21_vector_k_add_x_wrapper(int n, float k, PointerFloatWrapper x, int stride_x);

void math21_vector_kx_add_y_wrapper(int n, float k, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y,
                                    int stride_y);

void
math21_vector_normalize_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper mean, PointerFloatInputWrapper variance,
                                int mini_batch_size, int features_size,
                                int in_class_size);

void math21_vector_kx_with_in_class_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper k, int mini_batch_size,
                                            int features_size,
                                            int in_class_size);

void math21_vector_x_add_b_with_in_class_wrapper(PointerFloatWrapper x, PointerFloatInputWrapper b, int mini_batch_size,
                                                 int features_size,
                                                 int in_class_size);

float math21_vector_sum(const float *x, int n);

void math21_vector_sum_with_in_class_wrapper(PointerFloatWrapper db, PointerFloatInputWrapper dY, int mini_batch_size,
                                             int features_size,
                                             int in_class_size);

void math21_vector_sum_SchurProduct_with_in_class_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper dY,
                                                          int mini_batch_size, int features_size,
                                                          int in_class_size, PointerFloatWrapper dk);

void math21_vector_set_wrapper(int n, float value, PointerFloatWrapper X, int stride_x);

void math21_vector_set_int_wrapper(int n, int value, PointerIntWrapper X, int stride_x);

void math21_vector_assign_3d_d2_wrapper(PointerFloatInputWrapper data1, PointerFloatWrapper data2,
                                        int d1, int d2, int d3, int d2y, int offset2, int isToSmall);

void math21_vector_transpose_d1234_to_d1324_wrapper(PointerFloatInputWrapper x, PointerFloatWrapper y,
                                                    int d1, int d2, int d3, int d4);

void math21_vector_feature2d_add_2_wrapper(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_add_3_wrapper(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float kx2, PointerFloatInputWrapper X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_sample_wrapper(
        int mini_batch_size,
        PointerFloatWrapper X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k,
        PointerFloatWrapper Y);

void math21_vector_clip_wrapper(int n, float k, PointerFloatWrapper x, int stride);

void math21_vector_xy_wrapper(int n, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y, int stride_y);

void
math21_vector_assign_by_mask_wrapper(int n, PointerFloatWrapper x, float mask_num, PointerFloatInputWrapper mask,
                                     float val);

void math21_vector_kx_by_mask_wrapper(int n, float k, PointerFloatWrapper x, PointerFloatInputWrapper mask,
                                      float mask_num);

NumB math21_vector_isEmpty_wrapper(PointerVoidInputWrapper x);

PointerVoidWrapper math21_vector_getEmpty_wrapper();

PointerN8Wrapper math21_vector_getEmpty_N8_wrapper();

PointerFloatWrapper math21_vector_getEmpty_R32_wrapper();

void math21_vector_push_wrapper(PointerFloatWrapper x_gpu, const float *x, size_t n);

void math21_vector_push_N8_wrapper(PointerN8Wrapper x_gpu, const NumN8 *x, size_t n);

void math21_vector_pull_wrapper(PointerFloatInputWrapper x_gpu, float *x, size_t n);

void math21_vector_pull_N8_wrapper(PointerN8InputWrapper x_gpu, NumN8 *x, size_t n);

void math21_vector_log_pointer_wrapper(PointerFloatInputWrapper v);

void math21_vector_pr_rand_uniform_01_wrapper(PointerFloatWrapper v, int size);

void math21_vector_loss_l1_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                   PointerFloatWrapper dx, PointerFloatWrapper error);

void math21_vector_loss_l2_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                   PointerFloatWrapper dx, PointerFloatWrapper error);

void math21_vector_loss_smooth_l1_wrapper(int n, PointerFloatInputWrapper x, PointerFloatInputWrapper t,
                                          PointerFloatWrapper dx, PointerFloatWrapper error);

void math21_vector_zero_by_thresh_wrapper(int n, PointerFloatWrapper x, int stride_x, float thresh);

#ifdef __cplusplus
}
#endif
