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

#ifdef MATH21_FLAG_USE_CUDA

// v is created.
float *math21_vector_deserialize_c_cuda(FILE *f, size_t *n);

float *math21_vector_deserialize_from_file_cuda(const char *name, size_t *n);

void math21_vector_serialize_c_cuda(FILE *f, const float *v, size_t n);

void math21_vector_serialize_to_file_cuda(const char *name, const float *v, size_t n);

// print
void math21_vector_save_cuda(const char *name, const float *v, size_t from, size_t to);

void math21_vector_log_cuda(const char *name, const float *v, size_t from, size_t to);

float *math21_vector_create_with_default_value_cuda(size_t n, float value);

int *math21_vector_create_with_default_value_int_cuda(size_t n, int value);

void *math21_vector_create_buffer_cuda(size_t n, size_t elementSize);

float *math21_vector_resize_with_default_value_cuda(float *v, size_t n, float value);

void *math21_vector_resize_buffer_cuda(void *v, size_t n, size_t elementSize);

int *math21_vector_resize_with_default_value_int_cuda(int *v, size_t n, int value);

float *math21_vector_create_from_cpuvector_cuda(size_t n, const float *x, int stride_x);

int *math21_vector_create_from_cpuvector_int_cuda(size_t n, const int *x, int stride_x);

void math21_vector_free_cuda(void *x_gpu);

void
math21_vector_mean_fast_cuda(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean);

void math21_vector_mean_cuda(const float *X, int mini_batch_size, int features_size, int in_class_size, float *mean);

void math21_vector_variance_fast_cuda(const float *X, const float *mean, int mini_batch_size, int features_size,
                                      int in_class_size, float *variance);

void math21_vector_variance_cuda(const float *X, const float *mean, int mini_batch_size, int features_size,
                                 int in_class_size, float *variance);

void math21_vector_assign_from_vector_N8_cuda(int N, const NumN8 *X, NumN8 *Y);

void math21_vector_assign_from_vector_cuda(int N, const float *X, int INCX, float *Y, int stride_y);

void math21_vector_kx_cuda(int n, float k, float *x, int stride_x);

void math21_vector_k_add_x_cuda(int n, float k, float *x, int stride_x);

void math21_vector_kx_add_y_cuda(int n, float k, const float *x, int stride_x, float *y, int stride_y);

void
math21_vector_normalize_cuda(float *x, const float *mean, const float *variance, int mini_batch_size, int features_size,
                             int in_class_size);

void math21_vector_kx_with_in_class_cuda(float *x, const float *k, int mini_batch_size, int features_size,
                                         int in_class_size);

void math21_vector_x_add_b_with_in_class_cuda(float *x, const float *b, int mini_batch_size, int features_size,
                                              int in_class_size);

void math21_vector_sum_with_in_class_cuda(float *db, const float *dY, int mini_batch_size, int features_size,
                                          int in_class_size);

void math21_vector_sum_SchurProduct_with_in_class_cuda(const float *X, const float *dY, int mini_batch_size,
                                                       int features_size,
                                                       int in_class_size, float *dk);

void math21_vector_set_cuda(int N, float k, float *X, int stride_x);

void math21_vector_set_int_cuda(int n, int value, int *X, int stride_x);

void math21_vector_assign_3d_d2_cuda(const float *x, float *y,
                                     int d1, int d2, int d3, int d2y, int offset2, int isToSmall);

void math21_vector_transpose_d1234_to_d1324_cuda(const float *x, float *y, int d1, int d2, int d3, int d4);

void math21_vector_feature2d_add_2_cuda(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_add_3_cuda(
        int mini_batch_size,
        float kx, const float *X, int nch_X, int nr_X, int nc_X,
        float kx2, const float *X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, float *Y, int nch_Y, int nr_Y, int nc_Y);

void math21_vector_feature2d_sample_cuda(int mini_batch_size, float *X, int nch_X, int nr_X, int nc_X, int stride_X,
                                         int is_upsample, float k, float *Y);

void math21_vector_clip_cuda(int n, float k, float *x, int stride_x);

void math21_vector_xy_cuda(int n, const float *x, int stride_x, float *y, int stride_y);

void math21_vector_assign_by_mask_cuda(int n, float *x, float mask_num, const float *mask, float val);

void math21_vector_kx_by_mask_cuda(int n, float k, float *x, const float *mask, float mask_num);

void math21_vector_pr_rand_uniform_01_cuda(float *x_gpu, size_t n);

void math21_vector_loss_l1_cuda(int n, const float *x, const float *t, float *dx, float *error);

void math21_vector_loss_l2_cuda(int n, const float *x, const float *t, float *dx, float *error);

void math21_vector_loss_smooth_l1_cuda(int n, const float *x, const float *t, float *dx, float *error);

void math21_vector_zero_by_thresh_cuda(int n, float *x, int stride_x, float thresh);

#endif

#ifdef __cplusplus
}
#endif
