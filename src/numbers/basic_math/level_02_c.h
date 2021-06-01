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

#include "../config/config.h"
#include "../number_types.h"
#include "../export.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MATH21_IS_FROM_OPENCL)
#define MATH21_DEVICE_GLOBAL __global
#else
#define MATH21_DEVICE_GLOBAL
#endif


// see xj_index_1d_to_nd
// see math21_operator_number_index_1d_to_nd
// d is shape, index from 1
M21_EXPORT void math21_device_index_1d_to_nd(NumN *x, NumN y, MATH21_DEVICE_GLOBAL const NumN *d, NumN n);

// todo: maybe use M21_EXPORT
// d is shape, index from 1
M21_EXPORT void math21_device_index_nd_to_1d(const NumN *x, NumN *py, MATH21_DEVICE_GLOBAL const NumN *d, NumN n);

// see xj_index_1d_to_5d
M21_EXPORT void math21_device_index_1d_to_5d(NumN *x1, NumN *x2, NumN *x3, NumN *x4, NumN *x5, NumN y,
                                             NumN d1, NumN d2, NumN d3, NumN d4, NumN d5);

// see xj_index_5d_to_1d
M21_EXPORT void math21_device_index_5d_to_1d(NumN x1, NumN x2, NumN x3, NumN x4, NumN x5, NumN *py,
                                             NumN d1, NumN d2, NumN d3, NumN d4, NumN d5);

M21_EXPORT void math21_device_index_1d_to_4d(NumN *x1, NumN *x2, NumN *x3, NumN *x4, NumN y,
                                             NumN d1, NumN d2, NumN d3, NumN d4);

M21_EXPORT void math21_device_index_4d_to_1d(NumN x1, NumN x2, NumN x3, NumN x4, NumN *py,
                                             NumN d1, NumN d2, NumN d3, NumN d4);

M21_EXPORT void math21_device_index_1d_to_3d(NumN *x1, NumN *x2, NumN *x3, NumN y,
                                             NumN d1, NumN d2, NumN d3);

M21_EXPORT void math21_device_index_1d_to_3d_fast(NumN *x1, NumN *x2, NumN *x3,
                                                  NumN y, NumN d2, NumN d3);

M21_EXPORT void math21_device_index_3d_to_1d(NumN x1, NumN x2, NumN x3, NumN *py,
                                             NumN d1, NumN d2, NumN d3);

M21_EXPORT void math21_device_index_3d_to_1d_fast(NumN x1, NumN x2, NumN x3, NumN *py,
                                                  NumN d2, NumN d3);

M21_EXPORT NumN math21_device_image_get_1d_index(NumN ich, NumN ir, NumN ic,
                                                 NumN nch, NumN nr, NumN nc, NumB isInterleaved);

M21_EXPORT NumR math21_device_image_get_pixel(MATH21_DEVICE_GLOBAL const NumR *data, NumN ich, NumN ir, NumN ic,
                                              NumN nch, NumN nr, NumN nc, NumB isInterleaved);

M21_EXPORT NumB math21_device_image_get_pixel_bilinear_interpolate(
        MATH21_DEVICE_GLOBAL const NumR *data, NumR *value, NumN ich, NumR _ir, NumR _ic,
        NumN nch, NumN nr, NumN nc, NumB isInterleaved);

// deprecate, use template
M21_EXPORT NumB math21_device_image_get_pixel_bilinear_interpolate_32(
        MATH21_DEVICE_GLOBAL const NumR32 *data, NumR *value, NumN ich, NumR _ir, NumR _ic,
        NumN nch, NumN nr, NumN nc, NumB isInterleaved);

M21_EXPORT void math21_device_index_1d_to_2d(NumN *x1, NumN *x2, NumN y,
                                             NumN d1, NumN d2);

M21_EXPORT void math21_device_index_1d_to_2d_fast(NumN *x1, NumN *x2, NumN y,
                                                  NumN d2);

M21_EXPORT void math21_device_index_2d_to_1d(NumN x1, NumN x2, NumN *py,
                                             NumN d1, NumN d2);

M21_EXPORT void math21_device_index_2d_to_1d_fast(NumN x1, NumN x2, NumN *py,
                                                  NumN d2);

M21_EXPORT void math21_device_index_add_to_c_2(NumN n, const NumN *A, MATH21_DEVICE_GLOBAL const NumN *B, NumN *C);

M21_EXPORT void math21_device_broadcast_index_to_original_brackets(const NumN *index,
                                                                   MATH21_DEVICE_GLOBAL const NumN *d_ori,
                                                                   NumN *index_ori, NumN dims_ori);

// this is used to avoid the warning: pointer points outside of underlying object.
M21_EXPORT NumN *math21_device_pointer_NumN_decrease_one(NumN *p);


// see math21_operator_container_replace_inc
// replace A by R where A(i) = x.
M21_EXPORT void math21_device_index_replace_inc(NumN n, const NumN *A, NumN *B, const NumN *R, NumN x);

M21_EXPORT void math21_device_index_replace_inc_global_1(
        NumN n, MATH21_DEVICE_GLOBAL const NumN *A, NumN *B, const NumN *R, NumN x);

typedef NumR (*math21_type_f_min_like)(NumR value, NumR x, NumN i);

typedef NumR (*math21_type_f_argmin_like)(NumR value, NumR x, NumN *i_value, NumN i_x, NumN i);

typedef NumR (*math21_type_f_add_like)(NumR x, NumR y);

typedef NumR (*math21_type_f_sin_like)(NumR x);

//typedef NumR (*math21_type_f_kx_like)(NumR x, NumR y);
typedef math21_type_f_add_like math21_type_f_kx_like;

typedef NumR (*math21_type_f_inner_product_like)(NumR value, NumR x, NumR y, NumN i);

typedef NumR (*math21_type_f_addto_like)(NumR value, NumR x);

M21_EXPORT NumR math21_device_f_sum(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_norm1(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_norm2_square(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_mean(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_max(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_min(NumR value, NumR x, NumN i);

M21_EXPORT NumR math21_device_f_argmin(NumR value, NumR x, NumN *i_value, NumN i_x, NumN i);

M21_EXPORT NumR math21_device_f_argmax(NumR value, NumR x, NumN *i_value, NumN i_x, NumN i);

M21_EXPORT NumR math21_device_f_inner_product(NumR value, NumR x, NumR y, NumN i);

M21_EXPORT NumR math21_device_f_distance_1(NumR value, NumR x, NumR y, NumN i);

M21_EXPORT NumR math21_device_f_distance_2_square(NumR value, NumR x, NumR y, NumN i);


M21_EXPORT NumR math21_device_f_add(NumR x, NumR y);

M21_EXPORT NumR math21_device_f_subtract(NumR x, NumR y);

M21_EXPORT NumR math21_device_f_multiply(NumR x, NumR y);

M21_EXPORT NumR math21_device_f_divide(NumR x, NumR y);

M21_EXPORT NumR math21_device_f_is_equal(NumR x, NumR y);

M21_EXPORT NumR math21_device_f_sin(NumR x);

M21_EXPORT NumR math21_device_f_cos(NumR x);

M21_EXPORT NumR math21_device_f_tan(NumR x);

M21_EXPORT NumR math21_device_f_exp(NumR x);

M21_EXPORT NumR math21_device_f_log(NumR x);

M21_EXPORT NumR math21_device_f_abs(NumR x);

M21_EXPORT NumR math21_device_f_xk_subtract(NumR k, NumR x);

M21_EXPORT NumR math21_device_f_xk_divide(NumR k, NumR x);

M21_EXPORT NumR math21_device_f_kx_pow(NumR k, NumR x);

M21_EXPORT NumR math21_device_f_xk_pow(NumR k, NumR x);

M21_EXPORT NumR math21_device_f_addto(NumR value, NumR x);

M21_EXPORT NumR math21_device_f_multo(NumR value, NumR x);

#ifdef __cplusplus
}
#endif
