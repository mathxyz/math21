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
#include "01.h"

namespace math21 {
    template<typename T>
    T math21_op_get_num(const Tensor <T> &_x) {
        MATH21_ASSERT(_x.size() == 1);
        Tensor<T> x;
        x = _x;
        return x(1);
    }

    template<typename T>
    void math21_op_sqrt_onto(Tensor <T> &x);

    template<typename T>
    void math21_op_sum(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                       NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_sum, isKeepingDims);
    }

    template<typename T>
    void math21_op_norm(const Tensor <T> &x, Tensor <T> &y, NumN n, const VecN &axes,
                        NumB isKeepingDims = 0) {
        if (n == 1) {
            math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_norm1, isKeepingDims);
        } else if (n == 2) {
            math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_norm2_square, isKeepingDims);
            math21_op_sqrt_onto(y);
        } else {
            MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
        }
    }

    template<typename T>
    void math21_op_mean(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                        NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_mean, isKeepingDims);
    }

    template<typename T>
    void math21_op_max(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                       NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_max, isKeepingDims);
    }

    template<typename T>
    void math21_op_min(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                       NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_min, isKeepingDims);
    }

    template<typename T>
    void math21_op_argmax(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                          NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_argmax, isKeepingDims);
    }

    template<typename T>
    void math21_op_argmin(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                          NumB isKeepingDims = 0) {
        math21_op_tensor_f_shrink_using_axes(x, y, axes, m21_fname_argmin, isKeepingDims);
    }

    template<typename T>
    void math21_op_inner_product(
            const Tensor <T> &x1, const Tensor <T> &x2, Tensor <T> &y,
            const VecN &axes = VecN(), NumB isKeepingDims = 0) {
        math21_op_tensor_f_inner_product_like_shrink_using_axes(
                x1, x2, y, axes, m21_fname_inner_product, isKeepingDims);
    }

    template<typename T>
    void math21_op_distance_1(
            const Tensor <T> &x1, const Tensor <T> &x2, Tensor <T> &y,
            const VecN &axes, NumB isKeepingDims = 0) {
        math21_op_tensor_f_inner_product_like_shrink_using_axes(
                x1, x2, y, axes, m21_fname_distance_1, isKeepingDims);
    }

    template<typename T>
    void math21_op_distance_2_square(
            const Tensor <T> &x1, const Tensor <T> &x2, Tensor <T> &y,
            const VecN &axes, NumB isKeepingDims = 0) {
        math21_op_tensor_f_inner_product_like_shrink_using_axes(
                x1, x2, y, axes, m21_fname_distance_2_square, isKeepingDims);
    }

    template<typename T>
    void math21_op_add(const Tensor <T> &x1,
                       const Tensor <T> &x2,
                       Tensor <T> &y) {
        math21_op_tensor_f_with_broadcast(m21_fname_add, x1, x2, y);
    }

    template<typename T>
    void math21_op_add_onto_1(Tensor <T> &x1, const Tensor <T> &x2) {
        math21_op_tensor_f_onto_1_with_broadcast(m21_fname_add, x1, x2);
    }

    template<typename T>
    void math21_op_add_onto_2(const Tensor <T> &x1, Tensor <T> &x2) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_add, x1, x2);
    }

    template<typename T>
    void math21_op_subtract(const Tensor <T> &x1,
                            const Tensor <T> &x2,
                            Tensor <T> &y) {
        math21_op_tensor_f_with_broadcast(m21_fname_subtract, x1, x2, y);
    }

    template<typename T>
    void math21_op_subtract_onto_1(Tensor <T> &x1, const Tensor <T> &x2) {
        math21_op_tensor_f_onto_1_with_broadcast(m21_fname_subtract, x1, x2);
    }

    template<typename T>
    void math21_op_subtract_onto_2(const Tensor <T> &x1, Tensor <T> &x2) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_subtract, x1, x2);
    }

    template<typename T>
    void math21_op_mul(const Tensor <T> &x1,
                       const Tensor <T> &x2,
                       Tensor <T> &y) {
        math21_op_tensor_f_with_broadcast(m21_fname_multiply, x1, x2, y);
    }

    template<typename T>
    void math21_op_mul_onto_1(Tensor <T> &x1, const Tensor <T> &x2) {
        math21_op_tensor_f_onto_1_with_broadcast(m21_fname_multiply, x1, x2);
    }

    template<typename T>
    void math21_op_mul_onto_2(const Tensor <T> &x1, Tensor <T> &x2) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_multiply, x1, x2);
    }

    template<typename T>
    void math21_op_divide(const Tensor <T> &x1,
                          const Tensor <T> &x2,
                          Tensor <T> &y) {
        math21_op_tensor_f_with_broadcast(m21_fname_divide, x1, x2, y);
    }

    template<typename T>
    void math21_op_divide_onto_1(Tensor <T> &x1, const Tensor <T> &x2) {
        math21_op_tensor_f_onto_1_with_broadcast(m21_fname_divide, x1, x2);
    }

    template<typename T>
    void math21_op_divide_onto_2(const Tensor <T> &x1, Tensor <T> &x2) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_divide, x1, x2);
    }

    template<typename T>
    void math21_op_ele_is_equal(const Tensor <T> &x1,
                                const Tensor <T> &x2,
                                Tensor <T> &y) {
        math21_op_tensor_f_with_broadcast(m21_fname_ele_is_equal, x1, x2, y);
    }

    template<typename T>
    void math21_op_ele_is_equal_onto_1(Tensor <T> &x1, const Tensor <T> &x2) {
        math21_op_tensor_f_onto_1_with_broadcast(m21_fname_ele_is_equal, x1, x2);
    }

    template<typename T>
    void math21_op_ele_is_equal_onto_2(const Tensor <T> &x1, Tensor <T> &x2) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_ele_is_equal, x1, x2);
    }

    template<typename T>
    void math21_op_ele_is_equal(T k, const Tensor <T> &x1,
                                Tensor <T> &y) {
        Tensor<T> x2;
        x2.setDeviceType(x1.getDeviceType());
        x2.setSize(1);
        x2 = k;
        math21_op_tensor_f_with_broadcast(m21_fname_ele_is_equal, x1, x2, y);
    }

    // shape_x, shape_mask => shape_y if y is empty
    // shape_x, shape_mask must be compatible to shape_y respectively if y is not empty
    template<typename T>
    void math21_op_set_using_mask(const Tensor <T> &x, Tensor <T> &y,
                                  const Tensor <T> &mask) {
        math21_op_tensor_f_with_broadcast(m21_fname_set_using_mask, x, mask, y);
    }

    template<typename T>
    void math21_op_set_using_mask_onto_mask(const Tensor <T> &x, Tensor <T> &mask) {
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_set_using_mask, x, mask);
    }

    template<typename T>
    void math21_op_set_using_mask(NumR k, Tensor <T> &y,
                                  const Tensor <T> &mask) {
        Tensor<T> x;
        x.setDeviceType(mask.getDeviceType());
        x.setSize(1);
        x = (T) k;
        math21_op_set_using_mask(x, y, mask);
    }

    template<typename T>
    void math21_op_sin(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_sin, x, y);
    }

    template<typename T>
    void math21_op_sin_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_sin, x);
    }

    template<typename T>
    void math21_op_cos(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_cos, x, y);
    }

    template<typename T>
    void math21_op_cos_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_cos, x);
    }

    template<typename T>
    void math21_op_tan(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_tan, x, y);
    }

    template<typename T>
    void math21_op_tan_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_tan, x);
    }

    // here base is Euler's number
    template<typename T>
    void math21_op_exp(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_exp, x, y);
    }

    template<typename T>
    void math21_op_exp_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_exp, x);
    }

    // here base is Euler's number
    template<typename T>
    void math21_op_log(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_log, x, y);
    }

    template<typename T>
    void math21_op_log_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_log, x);
    }

    template<typename T>
    void math21_op_abs(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_sin_like(m21_fname_abs, x, y);
    }

    template<typename T>
    void math21_op_abs_onto(Tensor <T> &x) {
        math21_op_tensor_f_sin_like_onto(m21_fname_abs, x);
    }

    template<typename T>
    void math21_op_add(NumR k, const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_kx_add, k, x, y);
    }

    template<typename T>
    void math21_op_add_onto(NumR k, Tensor <T> &x) {
        math21_op_tensor_f_kx_like_onto(m21_fname_kx_add, k, x);
    }

    template<typename T>
    void math21_op_subtract(NumR k, const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_kx_subtract, k, x, y);
    }

    template<typename T>
    void math21_op_subtract_onto(NumR k, Tensor <T> &x) {
        math21_op_tensor_f_kx_like_onto(m21_fname_kx_subtract, k, x);
    }

    template<typename T>
    void math21_op_subtract(const Tensor <T> &x, NumR k, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_xk_subtract, k, x, y);
    }

    template<typename T>
    void math21_op_subtract_onto(Tensor <T> &x, NumR k) {
        math21_op_tensor_f_kx_like_onto(m21_fname_xk_subtract, k, x);
    }

    template<typename T>
    void math21_op_mul(NumR k, const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_kx_mul, k, x, y);
    }

    template<typename T>
    void math21_op_mul_onto(NumR k, Tensor <T> &x) {
        if (k != 1) {
            math21_op_tensor_f_kx_like_onto(m21_fname_kx_mul, k, x);
        }
    }

    template<typename T>
    void math21_op_divide(NumR k, const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_kx_divide, k, x, y);
    }

    template<typename T>
    void math21_op_divide_onto(NumR k, Tensor <T> &x) {
        math21_op_tensor_f_kx_like_onto(m21_fname_kx_divide, k, x);
    }

    template<typename T>
    void math21_op_divide(const Tensor <T> &x, NumR k, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_xk_divide, k, x, y);
    }

    template<typename T>
    void math21_op_divide_onto(Tensor <T> &x, NumR k) {
        math21_op_tensor_f_kx_like_onto(m21_fname_xk_divide, k, x);
    }

    template<typename T>
    void math21_op_pow(const Tensor <T> &x, NumR p, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_xk_pow, p, x, y);
    }

    template<typename T>
    void math21_op_square(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_xk_pow, 2, x, y);
    }

    template<typename T>
    void math21_op_sqrt(const Tensor <T> &x, Tensor <T> &y) {
        math21_op_tensor_f_kx_like(m21_fname_xk_pow, 0.5, x, y);
    }

    template<typename T>
    void math21_op_pow_onto(Tensor <T> &x, NumR p) {
        math21_op_tensor_f_kx_like_onto(m21_fname_xk_pow, p, x);
    }

    template<typename T>
    void math21_op_square_onto(Tensor <T> &x) {
        math21_op_tensor_f_kx_like_onto(m21_fname_xk_pow, 2, x);
    }

    template<typename T>
    void math21_op_sqrt_onto(Tensor <T> &x) {
        math21_op_tensor_f_kx_like_onto(m21_fname_xk_pow, 0.5, x);
    }

    // see math21_operator_tensor_f_along_axes(x, y, math21_operator_vector_logsumexp, axes, isKeepingDims);
    template<typename T>
    void math21_op_logsumexp(const Tensor <T> &x, Tensor <T> &y, const VecN &axes,
                             NumB isKeepingDims = 0) {
        VecN index;
        math21_operator_tensor_f_shrink_axes_to_index(x.dims(), axes, index);

        Tensor<T> max;
        math21_op_tensor_f_shrink(x, max, index, m21_fname_max, 1);
        Tensor<T> x0;
        math21_op_tensor_f_with_broadcast(m21_fname_subtract, x, max, x0);
        math21_op_tensor_f_sin_like_onto(m21_fname_exp, x0);
        math21_op_tensor_f_shrink(x0, y, index, m21_fname_sum, isKeepingDims);
        math21_op_tensor_f_sin_like_onto(m21_fname_log, y);
        if (!isKeepingDims) {
            max.reshape(y.shape());
        }
        // no bc
        math21_op_tensor_f_onto_2_with_broadcast(m21_fname_add, max, y);
    }

    template<typename T>
    NumB math21_op_check_is_nan(const Tensor <T> &x) {
        if (x.is_cpu()) {
            return math21_operator_check_container_is_nan(x);
        } else {
//            if (x.isEmpty()) {
//                return 0;
//            }
            Tensor<T> x2;
            return math21_operator_check_container_is_nan(
                    math21_operator_tensor_to_cpu(x, x2));
        }
    }

    template<typename T>
    T math21_op_vector_sum(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_sum(x, y, VecN(), 0);
        return math21_op_get_num(y);
    }

    template<typename T>
    NumR math21_op_vector_norm(const Tensor <T> &x, NumN n) {
        Tensor<T> y;
        math21_op_norm(x, y, n, VecN(), 0);
        return (NumR)math21_op_get_num(y);
    }

    // x/norm(x,n)
    template<typename T>
    void math21_op_vector_normalize(Tensor <T> &x, NumN n) {
        Tensor<T> y;
        math21_op_norm(x, y, n, VecN(), 0);
        NumR norm = (NumR)math21_op_get_num(y);
        math21_op_mul_onto(1 / norm, x);
    }

    template<typename T>
    NumR math21_op_vector_mean(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_mean(x, y, VecN(), 0);
        return (NumR)math21_op_get_num(y);
    }

    template<typename T>
    T math21_op_vector_max(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_max(x, y, VecN(), 0);
        return math21_op_get_num(y);
    }

    template<typename T>
    T math21_op_vector_min(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_min(x, y, VecN(), 0);
        return math21_op_get_num(y);
    }

    template<typename T>
    NumN math21_op_vector_argmax(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_argmax(x, y, VecN(), 0);
        return (NumN) math21_op_get_num(y);
    }

    template<typename T>
    NumN math21_op_vector_argmin(const Tensor <T> &x) {
        Tensor<T> y;
        math21_op_argmin(x, y, VecN(), 0);
        return (NumN) math21_op_get_num(y);
    }

}