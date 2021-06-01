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
#include "../kernels/generic_02.kl"

namespace math21 {
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_sum_p = &math21_device_f_sum;
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_norm1_p = &math21_device_f_norm1;
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_norm2_square_p = &math21_device_f_norm2_square;
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_mean_p = &math21_device_f_mean;
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_max_p = &math21_device_f_max;
    M21_EXPT_DEVICE math21_type_f_min_like math21_device_f_min_p = &math21_device_f_min;
    M21_EXPT_DEVICE math21_type_f_argmin_like math21_device_f_argmax_p = &math21_device_f_argmax;
    M21_EXPT_DEVICE math21_type_f_argmin_like math21_device_f_argmin_p = &math21_device_f_argmin;

    template<typename T>
    void math21_template_tensor_f_shrink_cuda(NumN fname, NumN n, const T *x, T *y,
                                              NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
                                              NumN nb, const NumN *b,
                                              NumN nv, NumN dims_v, const NumN *dv) {
        x -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        b -= 1;
        dv -= 1;
        // method 1
        // cudaMemcpyFromSymbol(&f, d_f, sizeof(math21_type_f_min_like));

        // method 2
        m21cudaCallableFunctionPointer<math21_type_f_min_like> f_min_like;
        m21cudaCallableFunctionPointer<math21_type_f_argmin_like> f_argmin_like;

        if (fname == m21_fname_sum) {
            f_min_like.set(&math21_device_f_sum_p);
        } else if (fname == m21_fname_norm1) {
            f_min_like.set(&math21_device_f_norm1_p);
        } else if (fname == m21_fname_norm2_square) {
            f_min_like.set(&math21_device_f_norm2_square_p);
        } else if (fname == m21_fname_mean) {
            f_min_like.set(&math21_device_f_mean_p);
        } else if (fname == m21_fname_max) {
            f_min_like.set(&math21_device_f_max_p);
        } else if (fname == m21_fname_min) {
            f_min_like.set(&math21_device_f_min_p);
        } else if (fname == m21_fname_argmax) {
            f_argmin_like.set(&math21_device_f_argmax_p);
        } else if (fname == m21_fname_argmin) {
            f_argmin_like.set(&math21_device_f_argmin_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        if (f_min_like.ptr) {
            math21_template_tensor_f_shrink_cuda_kernel
                    << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (
                    f_min_like.ptr, n, x, y,
                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
        } else {
            math21_template_tensor_f_shrink_cuda_kernel
                    << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> > (
                    f_argmin_like.ptr, n, x, y,
                            dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv);
        }
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    M21_EXPT_DEVICE math21_type_f_inner_product_like math21_device_f_inner_product_p = &math21_device_f_inner_product;
    M21_EXPT_DEVICE math21_type_f_inner_product_like math21_device_f_distance_1_p = &math21_device_f_distance_1;
    M21_EXPT_DEVICE math21_type_f_inner_product_like math21_device_f_distance_2_square_p = &math21_device_f_distance_2_square;

    template<typename T>
    void math21_template_tensor_f_inner_product_like_shrink_cuda(NumN fname, NumN n,
                                                                 const T *x1, const T *x2, T *y,
                                                                 NumN dims_x, const NumN *dx, NumN dims_y,
                                                                 const NumN *dy,
                                                                 NumN nb, const NumN *b,
                                                                 NumN nv, NumN dims_v, const NumN *dv) {
        x1 -= 1;
        x2 -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        b -= 1;
        dv -= 1;
        m21cudaCallableFunctionPointer<math21_type_f_inner_product_like> f;
        if (fname == m21_fname_inner_product) {
            f.set(&math21_device_f_inner_product_p);
        } else if (fname == m21_fname_distance_1) {
            f.set(&math21_device_f_distance_1_p);
        } else if (fname == m21_fname_distance_2_square) {
            f.set(&math21_device_f_distance_2_square_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        math21_template_tensor_f_inner_product_like_shrink_cuda_kernel << < math21_cuda_gridsize(n),
                MATH21_CUDA_BLOCK_SIZE >> >
                (f.ptr, n, x1, x2, y,
                        dims_x, dx, dims_y, dy, nb, b, nv, dims_v,
                        dv);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    M21_EXPT_DEVICE math21_type_f_add_like math21_device_f_add_p = &math21_device_f_add;
    M21_EXPT_DEVICE math21_type_f_add_like math21_device_f_subtract_p = &math21_device_f_subtract;
    M21_EXPT_DEVICE math21_type_f_add_like math21_device_f_multiply_p = &math21_device_f_multiply;
    M21_EXPT_DEVICE math21_type_f_add_like math21_device_f_divide_p = &math21_device_f_divide;
    M21_EXPT_DEVICE math21_type_f_add_like math21_device_f_is_equal_p = &math21_device_f_is_equal;

    // todo: use index 1 for x, y
    // a special kind of sub
    // x is sub-tensor of y
    template<typename T>
    void math21_template_tensor_f_with_broadcast_in_dn_cuda(NumN fname, NumN n,
                                                            const T *x1,
                                                            const T *x2,
                                                            T *y,
                                                            NumN dims_x1, const NumN *dx1,
                                                            NumN dims_x2, const NumN *dx2,
                                                            NumN dims_y, const NumN *dy) {
        x1 -= 1;
        x2 -= 1;
        y -= 1;
        dx1 -= 1;
        dx2 -= 1;
        dy -= 1;
        m21cudaCallableFunctionPointer<math21_type_f_add_like> f;
        if (fname == m21_fname_add) {
            f.set(&math21_device_f_add_p);
        } else if (fname == m21_fname_subtract) {
            f.set(&math21_device_f_subtract_p);
        } else if (fname == m21_fname_multiply) {
            f.set(&math21_device_f_multiply_p);
        } else if (fname == m21_fname_divide) {
            f.set(&math21_device_f_divide_p);
        } else if (fname == m21_fname_ele_is_equal) {
            f.set(&math21_device_f_is_equal_p);
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        if (fname == m21_fname_set_using_mask) {
            math21_template_tensor_set_using_mask_in_dn_cuda_kernel << < math21_cuda_gridsize(n),
                    MATH21_CUDA_BLOCK_SIZE >> >
                    (n,
                            x1, x2, y,
                            dims_x1, dx1,
                            dims_x2, dx2,
                            dims_y, dy);
        } else {
            math21_template_tensor_f_with_broadcast_in_dn_cuda_kernel << < math21_cuda_gridsize(n),
                    MATH21_CUDA_BLOCK_SIZE >> >
                    (f.ptr, n,
                            x1, x2, y,
                            dims_x1, dx1,
                            dims_x2, dx2,
                            dims_y, dy);
        }
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    // todo: use index 1 for x, y
    template<typename T>
    void math21_template_vector_f_add_like_cuda(NumN fname, NumN n,
                                                const T *x1,
                                                const T *x2,
                                                T *y) {
        x1 -= 1;
        x2 -= 1;
        y -= 1;
        m21cudaCallableFunctionPointer<math21_type_f_add_like> f;
        if (fname == m21_fname_add) {
            f.set(&math21_device_f_add_p);
        } else if (fname == m21_fname_subtract) {
            f.set(&math21_device_f_subtract_p);
        } else if (fname == m21_fname_multiply) {
            f.set(&math21_device_f_multiply_p);
        } else if (fname == m21_fname_divide) {
            f.set(&math21_device_f_divide_p);
        } else if (fname == m21_fname_ele_is_equal) {
            f.set(&math21_device_f_is_equal_p);
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        if (fname == m21_fname_set_using_mask) {
            math21_template_vector_set_using_mask_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                            (n, x1, x2, y);
        } else {
            math21_template_vector_f_add_like_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                        (f.ptr, n, x1, x2, y);
        }
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_sin_p = &math21_device_f_sin;
    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_cos_p = &math21_device_f_cos;
    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_tan_p = &math21_device_f_tan;
    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_exp_p = &math21_device_f_exp;
    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_log_p = &math21_device_f_log;
    M21_EXPT_DEVICE math21_type_f_sin_like math21_device_f_abs_p = &math21_device_f_abs;

    template<typename T>
    void math21_template_vector_f_sin_like_cuda(NumN fname, NumN n,
                                                const T *x, T *y) {
        x -= 1;
        y -= 1;
        m21cudaCallableFunctionPointer<math21_type_f_sin_like> f;
        if (fname == m21_fname_sin) {
            f.set(&math21_device_f_sin_p);
        } else if (fname == m21_fname_cos) {
            f.set(&math21_device_f_cos_p);
        } else if (fname == m21_fname_tan) {
            f.set(&math21_device_f_tan_p);
        } else if (fname == m21_fname_exp) {
            f.set(&math21_device_f_exp_p);
        } else if (fname == m21_fname_log) {
            f.set(&math21_device_f_log_p);
        } else if (fname == m21_fname_abs) {
            f.set(&math21_device_f_abs_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        math21_template_vector_f_sin_like_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                    (f.ptr, n, x, y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }

    M21_EXPT_DEVICE math21_type_f_kx_like math21_device_f_xk_subtract_p = &math21_device_f_xk_subtract;
    M21_EXPT_DEVICE math21_type_f_kx_like math21_device_f_xk_divide_p = &math21_device_f_xk_divide;
    M21_EXPT_DEVICE math21_type_f_kx_like math21_device_f_kx_pow_p = &math21_device_f_kx_pow;
    M21_EXPT_DEVICE math21_type_f_kx_like math21_device_f_xk_pow_p = &math21_device_f_xk_pow;

    template<typename T>
    void math21_template_vector_f_kx_like_cuda(NumN fname, NumN n, T k,
                                               const T *x, T *y) {
        x -= 1;
        y -= 1;
        m21cudaCallableFunctionPointer<math21_type_f_kx_like> f;
        if (fname == m21_fname_kx_add) {
            f.set(&math21_device_f_add_p);
        } else if (fname == m21_fname_kx_subtract) {
            f.set(&math21_device_f_subtract_p);
        } else if (fname == m21_fname_xk_subtract) {
            f.set(&math21_device_f_xk_subtract_p);
        } else if (fname == m21_fname_kx_mul) {
            f.set(&math21_device_f_multiply_p);
        } else if (fname == m21_fname_kx_divide) {
            f.set(&math21_device_f_divide_p);
        } else if (fname == m21_fname_xk_divide) {
            f.set(&math21_device_f_xk_divide_p);
        } else if (fname == m21_fname_kx_pow) {
            f.set(&math21_device_f_kx_pow_p);
        } else if (fname == m21_fname_xk_pow) {
            f.set(&math21_device_f_xk_pow_p);
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        math21_template_vector_f_kx_like_cuda_kernel << < math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >> >
                                                                                   (f.ptr, n, k, x, y);
        math21_cuda_check_error(cudaPeekAtLastError());
    }
}