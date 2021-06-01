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
#include "../kernels/generic_01.kl"
#include "../kernels/generic_01_vector_set.kl"

namespace math21 {

    // a special kind of sub, region sub.
    // x is sub-tensor of y
    template<typename T>
    void math21_template_tensor_sub_set_or_get_cuda(NumN n, T *x, T *y, NumN dims,
                                                    const NumN *dx, const NumN *dy,
                                                    const NumN *offset, NumB isGet) {
        math21_tool_assert(dims <= MATH21_KERNEL_ARRAY_MAX_LENGTH);
        x -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        offset -= 1;

        math21_template_tensor_sub_set_or_get_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                        (n, x, y, dims, dx, dy, offset, isGet);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    // x = k*x
    template<typename T>
    void math21_template_vector_kx_cuda(NumN n, T k, T *x, NumN stride_x) {
        math21_template_vector_kx_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                            (n, k, x, stride_x);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_kx_add_y_cuda(NumN n, T k, const T *x, NumN stride_x, T *y, NumN stride_y) {
        x -= 1;
        y -= 1;
        math21_template_vector_kx_add_y_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                  (n, k, x, stride_x, y, stride_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    // see math21_vector_assign_from_vector_byte_cuda
    template<typename T1, typename T2>
    void math21_template_vector_set_by_vector_cuda(NumN n, const T1 *x, NumN stride_x, T2 *y, NumN stride_y,
                                                   NumN offset_x, NumN offset_y) {
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        math21_template_vector_set_by_vector_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                       (n, x, stride_x, y, stride_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_matrix_set_by_matrix_cuda(NumN d1, NumN d2,
                                                   const T *x, NumN d1_x, NumN d2_x, NumN stride1_x, NumN stride2_x,
                                                   T *y, NumN d1_y, NumN d2_y, NumN stride1_y, NumN stride2_y,
                                                   NumN offset_x, NumN offset_y) {
        d2_x = stride1_x * d2_x; // stride absorbed into next dim, so stride will become 1.
        d2_y = stride1_y * d2_y;
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        NumN n = d1 * d2;
        math21_template_matrix_set_by_matrix_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                       (n, d2, x, d2_x, stride2_x, y, d2_y, stride2_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_tensor_3d_set_by_tensor_3d_cuda(NumN d1, NumN d2, NumN d3,
                                                         const T *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                         NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                         T *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                         NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                         NumN offset_x, NumN offset_y) {
        d2_x = stride1_x * d2_x;
        d2_y = stride1_y * d2_y;
        d3_x = stride2_x * d3_x;
        d3_y = stride2_y * d3_y;
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        NumN n = d1 * d2 * d3;
        math21_template_tensor_3d_set_by_tensor_3d_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                             (n, d2, d3, x, d2_x, d3_x, stride3_x, y, d2_y, d3_y, stride3_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    M21_EXPT_DEVICE math21_type_f_addto_like math21_device_f_addto_p = &math21_device_f_addto;
    M21_EXPT_DEVICE math21_type_f_addto_like math21_device_f_multo_p = &math21_device_f_multo;

    template<typename T>
    void math21_template_tensor_3d_f_set_by_tensor_3d_cuda(NumN fname, NumN d1, NumN d2, NumN d3,
                                                           const T *x, NumN d1_x, NumN d2_x, NumN d3_x,
                                                           NumN stride1_x, NumN stride2_x, NumN stride3_x,
                                                           T *y, NumN d1_y, NumN d2_y, NumN d3_y,
                                                           NumN stride1_y, NumN stride2_y, NumN stride3_y,
                                                           NumN offset_x, NumN offset_y) {
        d2_x = stride1_x * d2_x;
        d2_y = stride1_y * d2_y;
        d3_x = stride2_x * d3_x;
        d3_y = stride2_y * d3_y;
        x += offset_x;
        y += offset_y;

        x -= 1;
        y -= 1;
        NumN n = d1 * d2 * d3;
        m21cudaCallableFunctionPointer<math21_type_f_addto_like> f;
        if (fname == m21_fname_addto) {
            f.set(&math21_device_f_addto_p);
        } else if (fname == m21_fname_multo) {
            f.set(&math21_device_f_multo_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        math21_template_tensor_3d_f_set_by_tensor_3d_cuda_kernel << < math21_cuda_gridsize(n),
                MATH21_CUDA_BLOCK_SIZE >> >
                (f.ptr, n, d2, d3, x, d2_x, d3_x, stride3_x, y, d2_y, d3_y, stride3_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_set_by_value_cuda(NumN n, T value, T *x, NumN stride_x) {
        x -= 1;
        math21_template_vector_set_by_value_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                      (n, value, x, stride_x);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_xy_cuda(NumN n, const T *x, NumN stride_x, T *y, NumN stride_y) {
        x -= 1;
        y -= 1;
        math21_template_vector_xy_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                            (n, x, stride_x, y, stride_y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_sin_cuda(NumN n, const T *x, T *y) {
        math21_template_vector_sin_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                             (n, x, y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_cos_cuda(NumN n, const T *x, T *y) {
        math21_template_vector_cos_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                             (n, x, y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_addToC_cuda(NumN n, const T *A, const T *B, T *C) {
        math21_template_vector_addToC_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                (n, A, B, C);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_vector_mulToC_cuda(NumN n, const T *A, const T *B, T *C) {
        math21_template_vector_mulToC_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                (n, A, B, C);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    // todo: use index 1 for x, y
    // a special kind of sub
    // x is sub-tensor of y
    template<typename T>
    void math21_template_vector_broadcast_in_dn_cuda(NumN n, const T *x, T *y,
                                                     NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy) {
        x -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        math21_template_vector_broadcast_in_dn_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                         (n, x, y, dims_x, dx, dims_y, dy);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    template<typename T>
    void math21_template_optimization_adam_update_part_2_cuda(NumN n, T *x, const T *m, const T *v,
                                                              T beta1, T beta2, T alpha, T eps, NumN t) {
        x -= 1;
        m -= 1;
        v -= 1;
        math21_template_optimization_adam_update_part_2_cuda_kernel << < math21_cuda_gridsize(n),
                MATH21_CUDA_BLOCK_SIZE >> >
                (n, x, m, v, beta1, beta2, alpha, eps, t);
        math21_cuda_check_error(cudaPeekAtLastError());
    }
}