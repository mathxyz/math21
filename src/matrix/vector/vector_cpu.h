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

float *math21_vector_deserialize_c_cpu(FILE *f, size_t *n);

NumN8 *math21_vector_deserialize_byte_c_cpu(FILE *f, size_t *n);

NumN8 *math21_vector_raw_deserialize_byte_c_cpu(FILE *f, size_t n);

float *math21_vector_deserialize_from_file_cpu(const char *name, size_t *n);

NumN8 *math21_vector_deserialize_from_file_byte_cpu(const char *name, size_t *n);

NumN8 *math21_vector_raw_deserialize_from_file_byte_cpu(const char *name, size_t n);

void math21_vector_serialize_c_cpu(FILE *f, const float *v, size_t n);

void math21_vector_serialize_byte_c_cpu(FILE *f, const NumN8 *v, size_t n);

void math21_vector_raw_serialize_byte_c_cpu(FILE *f, const NumN8 *v, size_t n);

void math21_vector_serialize_to_file_cpu(const char *name, const float *v, size_t n);

void math21_vector_serialize_to_file_byte_cpu(const char *name, const NumN8 *v, size_t n);

void math21_vector_raw_serialize_to_file_byte_cpu(const char *name, const NumN8 *v, size_t n);

void math21_vector_save_cpu(const char *name, const float *v, size_t from, size_t to);

void math21_vector_log_cpu(const char *name, const float *v, size_t from, size_t to);

float *math21_vector_create_with_default_value_cpu(size_t n, float value);

NumN8 *math21_vector_create_with_default_value_byte_cpu(size_t n, NumN8 value);

void *math21_vector_create_buffer_cpu(size_t n, size_t elementSize);

void *math21_vector_setSize_buffer_cpu(void *v, size_t n, size_t elementSize);

void *math21_vector_copy_buffer_cpu(void *dst, const void *src, size_t n, size_t elementSize);

float *math21_vector_resize_with_default_value_cpu(float *v, size_t n, float value);

void *math21_vector_resize_buffer_cpu(void *v, size_t n, size_t elementSize);

int *math21_vector_resize_with_default_value_int_cpu(int *v, size_t n, int value);

float *math21_vector_create_from_cpuvector_cpu(size_t n, const float *x, int stride_x);

int *math21_vector_create_from_cpuvector_int_cpu(size_t n, const int *x, int stride_x);

NumN8 *math21_vector_create_from_cpuvector_byte_cpu(NumSize n, const NumN8 *x, int stride_x);

void math21_vector_free_cpu(void *x);

void math21_vector_mean_cpu(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean);

void
math21_vector_variance_cpu(const float *X, const float *mean, int mini_batch_size, int features_size, int in_class_size,
                           float *variance);

void math21_vector_assign_from_vector_N8_cpu(int n, const NumN8 *x, NumN8 *y);

void math21_vector_assign_from_vector_cpu(int n, const float *x, int stride_x, float *y, int stride_y);

void math21_vector_assign_from_vector_int_cpu(int n, const int *x, int stride_x, int *y, int stride_y);

void math21_vector_assign_from_vector_byte_cpu(NumSize n, const NumN8 *x, int stride_x, NumN8 *y, int stride_y);

void math21_vector_kx_cpu(int n, float k, float *x, int stride_x);

void math21_vector_k_add_x_cpu(int n, float k, float *x, int stride_x);

void math21_vector_kx_add_y_cpu(int n, float k, const float *x, int stride_x, float *y, int stride_y);

void
math21_vector_normalize_cpu(float *x, const float *mean, const float *variance, int mini_batch_size, int features_size,
                            int in_class_size);

void
math21_vector_kx_with_in_class_cpu(float *x, const float *k, int mini_batch_size, int features_size, int in_class_size);

void math21_vector_x_add_b_with_in_class_cpu(float *x, const float *b, int mini_batch_size, int features_size,
                                             int in_class_size);

float math21_vector_sum_cpu(const float *v, int n);

void math21_vector_sum_with_in_class_cpu(float *db, const float *dY, int mini_batch_size, int features_size,
                                         int in_class_size);

void math21_vector_sum_SchurProduct_with_in_class_cpu(const float *X, const float *dY, int mini_batch_size,
                                                      int features_size,
                                                      int in_class_size, float *dk);

void math21_vector_set_cpu(int n, float value, float *x, int stride);

void math21_vector_set_int_cpu(int n, int value, int *x, int stride);

void math21_vector_set_byte_cpu(int n, NumN8 value, NumN8 *x, int stride);

void math21_vector_set_random_cpu(int n, float *v);

void math21_vector_log_byte_cpu(int n, NumN8 *x, int stride);

void math21_vector_assign_3d_d2_cpu(const float *data1, float *data2,
                                    int d1, int d2, int d3, int d2y, int offset2, int isToSmall);

void math21_vector_transpose_d1234_to_d1324_cpu(const float *x, float *y, int d1, int d2, int d3, int d4);

void math21_vector_feature2d_add_2_cpu(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_add_3_cpu(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float kx2, const float *X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_sample_cpu(
        int mini_batch_size,
        float *X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k, float *Y);

void math21_vector_clip_cpu(int n, float k, float *x, int stride);

void math21_vector_xy_cpu(int n, const float *x, int stride_x, float *y, int stride_y);

void math21_vector_pr_rand_uniform_01_cpu(float *v, int size);

void math21_vector_loss_l1_cpu(int n, const float *x, const float *t, float *dx, float *error);

void math21_vector_loss_l2_cpu(int n, const float *x, const float *t, float *dx, float *error);

void math21_vector_loss_smooth_l1_cpu(int n, const float *x, const float *t, float *dx, float *error);

#ifdef __cplusplus
}
#endif
