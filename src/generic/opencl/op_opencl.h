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

#ifdef MATH21_FLAG_USE_OPENCL

void math21_generic_tensor_sub_set_or_get_opencl(NumN n, PointerVoidWrapper x, PointerVoidWrapper y, NumN dims,
                                                 PointerNumNInputWrapper dx, PointerNumNInputWrapper dy,
                                                 PointerNumNInputWrapper offset, NumB isGet, NumN type);

void math21_generic_vector_kx_opencl(NumN n, NumR k, PointerVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_kx_add_y_opencl(NumN n, NumR k, PointerVoidInputWrapper x, NumN stride_x,
                                           PointerVoidWrapper y, NumN stride_y, NumN type);

void math21_generic_vector_set_by_vector_opencl(
        NumN n, PointerVoidInputWrapper x, NumN stride_x, PointerVoidWrapper y, NumN stride_y,
        NumN offset_x, NumN offset_y, NumN type1, NumN type2);

void math21_generic_matrix_set_by_matrix_opencl(NumN d1, NumN d2,
                                                PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN stride1_x,
                                                NumN stride2_x,
                                                PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN stride1_y,
                                                NumN stride2_y,
                                                NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_set_by_tensor_3d_opencl(NumN d1, NumN d2, NumN d3,
                                                      PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                      NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                      PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                      NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                      NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_f_set_by_tensor_3d_opencl(NumN fname, NumN d1, NumN d2, NumN d3,
                                                        PointerVoidInputWrapper x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                        NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                        PointerVoidWrapper y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                        NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                        NumN offset_x, NumN offset_y, NumN type);

void math21_generic_vector_set_by_value_opencl(NumN n, NumR value, PointerVoidWrapper x, NumN stride_x, NumN type);

void math21_generic_vector_xy_opencl(NumN n, PointerVoidInputWrapper x, NumN stride_x,
                                     PointerVoidWrapper y, NumN stride_y, NumN type);

void math21_generic_vector_sin_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y, NumN type);

void math21_generic_vector_cos_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y, NumN type);

void
math21_generic_vector_addToC_opencl(NumN n, PointerVoidInputWrapper A, PointerVoidInputWrapper B, PointerVoidWrapper C,
                                    NumN type);

void
math21_generic_vector_mulToC_opencl(NumN n, PointerVoidInputWrapper A, PointerVoidInputWrapper B, PointerVoidWrapper C,
                                    NumN type);

void math21_generic_broadcast_in_dn_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                           NumN dims_x, PointerNumNInputWrapper dx,
                                           NumN dims_y, PointerNumNInputWrapper dy,
                                           NumN type);

void math21_generic_optimization_adam_update_part_2_opencl(
        NumN x_size, PointerVoidWrapper x, PointerVoidInputWrapper m, PointerVoidInputWrapper v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type);

void math21_generic_tensor_f_shrink_opencl(NumN fname, NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                           NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y,
                                           PointerNumNInputWrapper dy,
                                           NumN nb, PointerNumNInputWrapper b,
                                           NumN nv, NumN dims_v, PointerNumNInputWrapper dv, NumN type);

void math21_generic_tensor_f_inner_product_like_shrink_opencl(
        NumN fname, NumN n,
        PointerVoidInputWrapper x1, PointerVoidInputWrapper x2, PointerVoidWrapper y,
        NumN dims_x, PointerNumNInputWrapper dx, NumN dims_y, PointerNumNInputWrapper dy,
        NumN nb, PointerNumNInputWrapper b,
        NumN nv, NumN dims_v, PointerNumNInputWrapper dv, NumN type);

void math21_generic_tensor_f_with_broadcast_in_dn_opencl(NumN fname, NumN n,
                                                         PointerVoidInputWrapper x1,
                                                         PointerVoidInputWrapper x2,
                                                         PointerVoidWrapper y,
                                                         NumN dims_x1, PointerNumNInputWrapper dx1,
                                                         NumN dims_x2, PointerNumNInputWrapper dx2,
                                                         NumN dims_y, PointerNumNInputWrapper dy, NumN type);

void math21_generic_vector_f_add_like_opencl(NumN fname, NumN n,
                                             PointerVoidInputWrapper x1,
                                             PointerVoidInputWrapper x2,
                                             PointerVoidWrapper y, NumN type);

void math21_generic_vector_f_sin_like_opencl(NumN fname, NumN n,
                                             PointerVoidInputWrapper x1,
                                             PointerVoidWrapper y, NumN type);


void math21_generic_vector_f_kx_like_opencl(NumN fname, NumN n,
                                            NumR k,
                                            PointerVoidInputWrapper x1,
                                            PointerVoidWrapper y, NumN type);

void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_opencl(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        PointerVoidInputWrapper A, NumN stride_a,
        PointerVoidInputWrapper B, NumN stride_b,
        NumR k2, PointerVoidWrapper C, NumN stride_c, NumN type);

void math21_generic_matrix_transpose_opencl(NumN n, PointerVoidInputWrapper x, PointerVoidWrapper y,
                                            NumN nr_x, NumN nc_x, NumN type1, NumN type2);

void math21_generic_tensor_swap_axes_24_in_d5_opencl(PointerVoidInputWrapper x, PointerVoidWrapper y,
                                            NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2);

#endif

#ifdef __cplusplus
}
#endif
