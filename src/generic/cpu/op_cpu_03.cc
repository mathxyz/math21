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

#include "template_cpu_03.h"
#include "op_cpu.h"

using namespace math21;


// here stride_a is trailing dimension of a
// stride1_abs = stride1 * stride = 1 * stride = stride
void math21_generic_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
        NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, NumR k1,
        const void *A, NumN stride_a,
        const void *B, NumN stride_b,
        NumR k2, void *C, NumN stride_c, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
                ta, tb, nr_C, nc_C, n_common, (NumR) k1, (const NumR *) A, stride_a,
                (const NumR *) B, stride_b, (NumR) k2, (NumR *) C, stride_c);
    } else if (type == m21_type_NumR32) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
                ta, tb, nr_C, nc_C, n_common, (NumR32) k1, (const NumR32 *) A, stride_a,
                (const NumR32 *) B, stride_b, (NumR32) k2, (NumR32 *) C, stride_c);
    } else if (type == m21_type_NumR64) {
        math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_cpu(
                ta, tb, nr_C, nc_C, n_common, (NumR64) k1, (const NumR64 *) A, stride_a,
                (const NumR64 *) B, stride_b, (NumR64) k2, (NumR64 *) C, stride_c);
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_matrix_transpose_cpu(NumN n, const void *x, void *y,
                                         NumN nr_x, NumN nc_x, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_transpose_cpu(n, (const NumN8 *) x, (NumN8 *) y, nr_x, nc_x);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_transpose_cpu(n, (const NumN8 *) x, (NumR *) y, nr_x, nc_x);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_matrix_transpose_cpu(n, (const NumR *) x, (NumN8 *) y, nr_x, nc_x);
        } else if (type2 == m21_type_NumN) {
            math21_template_matrix_transpose_cpu(n, (const NumR *) x, (NumN *) y, nr_x, nc_x);
        } else if (type2 == m21_type_NumR) {
            math21_template_matrix_transpose_cpu(n, (const NumR *) x, (NumR *) y, nr_x, nc_x);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}

void math21_generic_tensor_swap_axes_24_in_d5_cpu(const void *x, void *y,
                                                  NumN d1, NumN d2, NumN d3, NumN d4, NumN d5, NumN type1, NumN type2) {
    if (type1 == m21_type_NumN8) {
        if (type2 == m21_type_NumN8) {
            math21_template_tensor_swap_axes_24_in_d5_cpu((const NumN8 *) x, (NumN8 *) y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumR) {
            math21_template_tensor_swap_axes_24_in_d5_cpu((const NumN8 *) x, (NumR *) y, d1, d2, d3, d4, d5);
        } else {
            math21_tool_assert(0);
        }
    } else if (type1 == m21_type_NumR) {
        if (type2 == m21_type_NumN8) {
            math21_template_tensor_swap_axes_24_in_d5_cpu((const NumR *) x, (NumN8 *) y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumN) {
            math21_template_tensor_swap_axes_24_in_d5_cpu((const NumR *) x, (NumN *) y, d1, d2, d3, d4, d5);
        } else if (type2 == m21_type_NumR) {
            math21_template_tensor_swap_axes_24_in_d5_cpu((const NumR *) x, (NumR *) y, d1, d2, d3, d4, d5);
        } else {
            math21_tool_assert(0);
        }
    } else {
        math21_tool_assert(0);
    }
}
