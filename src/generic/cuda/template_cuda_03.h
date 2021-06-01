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

#include "inner.h"
#include "../../gpu/files.h"
#include "common.h"
#include "../kernels/generic_03.kl"
#include "../kernels/generic_03_transpose.kl"

namespace math21 {

// see math21_matrix_multiply_k1AB_add_k2C_similar
// C = k1*(A*B) + k2*C or similar
    template<typename T>
    void math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_cuda(
            NumB ta, NumB tb, NumN nr_C, NumN nc_C, NumN n_common, T k1,
            const T *A, NumN stride_a,
            const T *B, NumN stride_b,
            T k2, T *C, NumN stride_c) {
        A -= 1;
        B -= 1;
        C -= 1;
        NumN size = nr_C * nc_C;
        if (!ta && !tb) {
            math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nn_naive_cuda_kernel << <
            math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                        (size, nr_C, nc_C, n_common, k1, A, stride_a, B, stride_b, k2, C, stride_c);
        } else if (ta && !tb) {
            math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tn_naive_cuda_kernel << <
            math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                        (size, nr_C, nc_C, n_common, k1, A, stride_a, B, stride_b, k2, C, stride_c);
        } else if (!ta && tb) {
            math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_nt_naive_cuda_kernel << <
            math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                        (size, nr_C, nc_C, n_common, k1, A, stride_a, B, stride_b, k2, C, stride_c);
        } else {
            math21_template_matrix_multiply_onto_k1AB_add_k2C_similar_tt_naive_cuda_kernel << <
            math21_cuda_gridsize(size), MATH21_CUDA_BLOCK_SIZE >> >
                                        (size, nr_C, nc_C, n_common, k1, A, stride_a, B, stride_b, k2, C, stride_c);
        }
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T1, typename T2>
    void math21_template_matrix_transpose_cuda(NumN n, const T1 *x, T2 *y, NumN d1, NumN d2) {
        x -= 1;
        y -= 1;
        math21_template_matrix_transpose_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                            (n, x, y, d1, d2);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T1, typename T2>
    void math21_template_tensor_swap_axes_24_in_d5_cuda(const T1 *x, T2 *y, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        x -= 1;
        y -= 1;
        NumN n = d1 * d2 * d3 * d4 * d5;
        math21_template_tensor_swap_axes_24_in_d5_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                   (n, x, y, d1, d2, d3, d4, d5);
        math21_cuda_check_error(cudaPeekAtLastError());
    }
}