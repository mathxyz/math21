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


void math21_generic_tensor_sub_set_or_get_cuda(NumN n, void *x, void *y, NumN dims,
                                              const NumN *dx, const NumN *dy,
                                              const NumN *offset, NumB isGet, NumN type);

void math21_generic_vector_kx_cuda(NumN n, NumR k, void *x, NumN stride_x, NumN type);

void math21_generic_vector_kx_add_y_cuda(NumN n, NumR k, const void *x, NumN stride_x,
                                        void *y, NumN stride_y, NumN type);

void math21_generic_vector_set_by_vector_cuda(
        NumN n, const void *x, NumN stride_x, void *y, NumN stride_y,
        NumN offset_x, NumN offset_y, NumN type1, NumN type2);

void math21_generic_matrix_set_by_matrix_cuda(NumN d1, NumN d2,
                                             const void *x, NumN d1_x, NumN d2_x, NumN stride1_x, NumN stride2_x,
                                             void *y, NumN d1_y, NumN d2_y, NumN stride1_y, NumN stride2_y,
                                             NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_set_by_tensor_3d_cuda(NumN d1, NumN d2, NumN d3,
                                                   const void *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                   NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                   void *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                   NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                   NumN offset_x, NumN offset_y, NumN type);

void math21_generic_tensor_3d_f_set_by_tensor_3d_cuda(NumN fname, NumN d1, NumN d2, NumN d3,
                                                     const void *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                     NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                     void *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                     NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                     NumN offset_x, NumN offset_y, NumN type);

void math21_generic_vector_set_by_value_cuda(NumN n, NumR value, void *x, NumN stride_x, NumN type);

void math21_generic_vector_xy_cuda(NumN n, const void *x, NumN stride_x,
                                  void *y, NumN stride_y, NumN type);

void math21_generic_vector_sin_cuda(NumN n, const void *x, void *y, NumN type);

void math21_generic_vector_cos_cuda(NumN n, const void *x, void *y, NumN type);

void math21_generic_vector_addToC_cuda(NumN n, const void *A, const void *B, void *C, NumN type);

void math21_generic_vector_mulToC_cuda(NumN n, const void *A, const void *B, void *C, NumN type);

void math21_generic_broadcast_in_dn_cuda(NumN n, const void *x, void *y,
                                        NumN dims_x, const NumN *dx,
                                        NumN dims_y, const NumN *dy,
                                        NumN type);

void math21_generic_optimization_adam_update_part_2_cuda(
        NumN x_size, void *x, const void *m, const void *v,
        NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t, NumN type);

void math21_generic_tensor_f_shrink_cuda(NumN fname, NumN n, const void *x, void *y,
                                        NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
                                        NumN nb, const NumN *b,
                                        NumN nv, NumN dims_v, const NumN *dv, NumN type);

void math21_generic_tensor_f_inner_product_like_shrink_cuda(
        NumN fname, NumN n,
        const void *x1, const void *x2, void *y,
        NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
        NumN nb, const NumN *b,
        NumN nv, NumN dims_v, const NumN *dv, NumN type);

void math21_generic_tensor_f_with_broadcast_in_dn_cuda(NumN fname, NumN n,
                                                      const void *x1,
                                                      const void *x2,
                                                      void *y,
                                                      NumN dims_x1, const NumN *dx1,
                                                      NumN dims_x2, const NumN *dx2,
                                                      NumN dims_y, const NumN *dy, NumN type);

void math21_generic_vector_f_add_like_cuda(NumN fname, NumN n,
                                          const void *x1,
                                          const void *x2,
                                          void *y, NumN type);

void math21_generic_vector_f_sin_like_cuda(NumN fname, NumN n,
                                          const void *x1,
                                          void *y, NumN type);


void math21_generic_vector_f_kx_like_cuda(NumN fname, NumN n,
                                         NumR k,
                                         const void *x1,
                                         void *y, NumN type);

void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_cuda(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        const void *A, NumN stride_a,
        const void *B, NumN stride_b,
        NumR k2, void *C, NumN stride_c, NumN type);

void math21_generic_matrix_transpose_cuda(NumN n, const void *x, void *y,
                                         NumN nr_x, NumN nc_x, NumN type1, NumN type2);

void math21_generic_tensor_swap_axes_24_in_d5_cuda(const void *x, void *y,
                                                   NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1,
                                                   NumN type2);

#endif

#ifdef __cplusplus
}
#endif
