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

void math21_generic_tensor_sub_set_or_get_wrapper(NumN n, PointerVoidWrapper x, PointerVoidWrapper y, NumN dims,
                                                  PointerNumNInputWrapper dx, PointerNumNInputWrapper dy,
                                                  PointerNumNInputWrapper offset, NumB isGet, NumN type);

void math21_generic_vector_kx_wrapper(NumN n, NumR k, PointerVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_kx_add_y_wrapper(
        NumN n, NumR k, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y,
        NumN stride_y, NumN type);

void math21_generic_vector_set_by_vector_wrapper(
        NumN n, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y,
        NumN stride_y, NumN offset_x, NumN offset_y, NumN type1, NumN type2);


void math21_generic_matrix_set_by_matrix_wrapper(NumN d1, NumN d2,
                                                 PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                 NumN stride2_x,
                                                 PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                 NumN stride2_y,
                                                 NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_set_by_tensor_3d_wrapper(NumN d1, NumN d2, NumN d3,
                                                       PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                       NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                       PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                       NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                       NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_f_set_by_tensor_3d_wrapper(NumN fname, NumN d1, NumN d2, NumN d3,
                                                         PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                         NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                         PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                         NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                         NumN offset_x, NumN offset_y, NumN type);

void math21_generic_vector_set_by_value_wrapper(
        NumN n, NumR value, PointerVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_xy_wrapper(
        NumN n, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y,
        NumN stride_y, NumN type);

void math21_generic_vector_sin_wrapper(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y, NumN type);

void math21_generic_vector_cos_wrapper(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y, NumN type);

void math21_generic_vector_addToC_wrapper(NumN n, PointerVoidInputWrapper A,
                                          PointerVoidInputWrapper B, PointerVoidWrapper C, NumN type);

void math21_generic_vector_mulToC_wrapper(NumN n, PointerVoidInputWrapper A,
                                          PointerVoidInputWrapper B, PointerVoidWrapper C, NumN type);

void math21_generic_broadcast_in_dn_wrapper(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                            NumN dims_x, PointerNumNInputWrapper dx,
                                            NumN dims_y, PointerNumNInputWrapper dy,
                                            NumN type);

void math21_generic_optimization_adam_update_part_2_wrapper(
        NumN x_size, PointerVoidWrapper x, PointerVoidInputWrapper m, PointerVoidInputWrapper v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type);

void math21_generic_tensor_f_shrink_wrapper(
        NumN fname, NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
        NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y, PointerNumNInputWrapper dy,
        NumN n_b, PointerNumNInputWrapper b,
        NumN n_v, NumN dims_v, PointerNumNInputWrapper dv, NumN type);

void math21_generic_tensor_f_inner_product_like_shrink_wrapper(
        NumN fname, NumN n,
        PointerVoidInputWrapper x1, PointerVoidInputWrapper x2, PointerVoidWrapper y,
        NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y, PointerNumNInputWrapper dy,
        NumN n_b, PointerNumNInputWrapper b,
        NumN n_v, NumN dims_v, PointerNumNInputWrapper dv, NumN type);

void math21_generic_tensor_f_with_broadcast_in_dn_wrapper(NumN fname, NumN n,
                                                          PointerVoidInputWrapper x1,
                                                          PointerVoidInputWrapper x2,
                                                          PointerVoidWrapper y,
                                                          NumN dims_x1, PointerNumNInputWrapper dx1,
                                                          NumN dims_x2, PointerNumNInputWrapper dx2,
                                                          NumN dims_y, PointerNumNInputWrapper dy, NumN type);

void math21_generic_vector_f_add_like_wrapper(NumN fname, NumN n,
                                              PointerVoidInputWrapper x1,
                                              PointerVoidInputWrapper x2,
                                              PointerVoidWrapper y, NumN type);

void math21_generic_vector_f_sin_like_wrapper(NumN fname, NumN n,
                                              PointerVoidInputWrapper x1,
                                              PointerVoidWrapper y, NumN type);

void math21_generic_vector_f_kx_like_wrapper(NumN fname, NumN n, NumR k,
                                             PointerVoidInputWrapper x1,
                                             PointerVoidWrapper y, NumN type);

void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_wrapper(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        PointerVoidInputWrapper A, NumN stride_a,
        PointerVoidInputWrapper B, NumN stride_b,
        NumR k2, PointerVoidWrapper C, NumN stride_c, NumN type);

void math21_generic_matrix_transpose_wrapper(NumN n,
                                             PointerVoidInputWrapper x,
                                             PointerVoidWrapper y,
                                             NumN nr_x, NumN nc_x, NumN type1, NumN type2);

void math21_generic_tensor_swap_axes_24_in_d5_wrapper(
        PointerVoidInputWrapper x,
        PointerVoidWrapper y,
        NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2);

#ifdef __cplusplus
}
#endif
