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

#define MATH21_IS_FROM_CPU

#include "../kernels/generic_02.kl"

#undef MATH21_IS_FROM_CPU

namespace math21 {

    template<typename T>
    void math21_template_tensor_f_shrink_cpu(NumN fname, NumN n, const T *x, T *y,
                                             NumN dims_x, const NumN *dx, NumN dims_y, const NumN *dy,
                                             NumN nb, const NumN *b,
                                             NumN nv, NumN dims_v, const NumN *dv) {
        x -= 1;
        y -= 1;
        dx -= 1;
        dy -= 1;
        b -= 1;
        dv -= 1;
        NumN id;
        math21_type_f_min_like f_min_like = NULL;
        math21_type_f_argmin_like f_argmin_like = NULL;
        if (fname == m21_fname_sum) {
            f_min_like = math21_device_f_sum;
        } else if (fname == m21_fname_norm1) {
            f_min_like = math21_device_f_norm1;
        } else if (fname == m21_fname_norm2_square) {
            f_min_like = math21_device_f_norm2_square;
        } else if (fname == m21_fname_mean) {
            f_min_like = math21_device_f_mean;
        } else if (fname == m21_fname_max) {
            f_min_like = math21_device_f_max;
        } else if (fname == m21_fname_min) {
            f_min_like = math21_device_f_min;
        } else if (fname == m21_fname_argmax) {
            f_argmin_like = math21_device_f_argmax;
        } else if (fname == m21_fname_argmin) {
            f_argmin_like = math21_device_f_argmin;
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        if (f_min_like) {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_tensor_f_shrink_cpu_kernel(f_min_like, n, x, y,
                                                           dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv, id);
            }
        } else {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_tensor_f_shrink_cpu_kernel(f_argmin_like, n, x, y,
                                                           dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv,
                                                           id);
            }
        }

    }

    template<typename T>
    void math21_template_tensor_f_inner_product_like_shrink_cpu(NumN fname, NumN n,
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
        NumN id;
        math21_type_f_inner_product_like f = NULL;
        if (fname == m21_fname_inner_product) {
            f = math21_device_f_inner_product;
        } else if (fname == m21_fname_distance_1) {
            f = math21_device_f_distance_1;
        } else if (fname == m21_fname_distance_2_square) {
            f = math21_device_f_distance_2_square;
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_tensor_f_inner_product_like_shrink_cpu_kernel(f, n, x1, x2, y,
                                                                          dims_x, dx, dims_y, dy, nb, b, nv, dims_v, dv,
                                                                          id);
        }
    }

    // todo: use index 1 for x, y
    // a special kind of sub
    // x is sub-tensor of y
    template<typename T>
    void math21_template_tensor_f_with_broadcast_in_dn_cpu(NumN fname, NumN n,
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
        NumN id;
        math21_type_f_add_like f_add_like = NULL;
        if (fname == m21_fname_add) {
            f_add_like = math21_device_f_add;
        } else if (fname == m21_fname_subtract) {
            f_add_like = math21_device_f_subtract;
        } else if (fname == m21_fname_multiply) {
            f_add_like = math21_device_f_multiply;
        } else if (fname == m21_fname_divide) {
            f_add_like = math21_device_f_divide;
        } else if (fname == m21_fname_ele_is_equal) {
            f_add_like = math21_device_f_is_equal;
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
        if (fname == m21_fname_set_using_mask) {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_tensor_set_using_mask_in_dn_cpu_kernel(n,
                                                                       x1, x2, y,
                                                                       dims_x1, dx1,
                                                                       dims_x2, dx2,
                                                                       dims_y, dy, id);
            }
        } else {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_tensor_f_with_broadcast_in_dn_cpu_kernel(f_add_like, n,
                                                                         x1, x2, y,
                                                                         dims_x1, dx1,
                                                                         dims_x2, dx2,
                                                                         dims_y, dy, id);
            }
        }
    }

    // todo: use index 1 for x, y
    template<typename T>
    void math21_template_vector_f_add_like_cpu(NumN fname, NumN n,
                                               const T *x1,
                                               const T *x2,
                                               T *y) {
        x1 -= 1;
        x2 -= 1;
        y -= 1;
        NumN id;
        math21_type_f_add_like f_add_like = NULL;
        if (fname == m21_fname_add) {
            f_add_like = math21_device_f_add;
        } else if (fname == m21_fname_subtract) {
            f_add_like = math21_device_f_subtract;
        } else if (fname == m21_fname_multiply) {
            f_add_like = math21_device_f_multiply;
        } else if (fname == m21_fname_divide) {
            f_add_like = math21_device_f_divide;
        } else if (fname == m21_fname_ele_is_equal) {
            f_add_like = math21_device_f_is_equal;
        } else if (fname == m21_fname_set_using_mask) {
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }

        if (fname == m21_fname_set_using_mask) {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_vector_set_using_mask_cpu_kernel(n, x1, x2, y, id);
            }
        } else {
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_template_vector_f_add_like_cpu_kernel(f_add_like, n,
                                                             x1, x2, y, id);
            }
        }
    }

    template<typename T>
    void math21_template_vector_f_sin_like_cpu(NumN fname, NumN n,
                                               const T *x, T *y) {
        x -= 1;
        y -= 1;
        NumN id;
        math21_type_f_sin_like f = NULL;
        if (fname == m21_fname_sin) {
            f = math21_device_f_sin;
        } else if (fname == m21_fname_cos) {
            f = math21_device_f_cos;
        } else if (fname == m21_fname_tan) {
            f = math21_device_f_tan;
        } else if (fname == m21_fname_exp) {
            f = math21_device_f_exp;
        } else if (fname == m21_fname_log) {
            f = math21_device_f_log;
        } else if (fname == m21_fname_abs) {
            f = math21_device_f_abs;
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_vector_f_sin_like_cpu_kernel(f, n, x, y, id);
        }
    }

    template<typename T>
    void math21_template_vector_f_kx_like_cpu(NumN fname, NumN n, T k,
                                              const T *x, T *y) {
        x -= 1;
        y -= 1;
        NumN id;
        math21_type_f_kx_like f = NULL;
        if (fname == m21_fname_kx_add) {
            f = math21_device_f_add;
        } else if (fname == m21_fname_kx_subtract) {
            f = math21_device_f_subtract;
        } else if (fname == m21_fname_xk_subtract) {
            f = math21_device_f_xk_subtract;
        } else if (fname == m21_fname_kx_mul) {
            f = math21_device_f_multiply;
        } else if (fname == m21_fname_kx_divide) {
            f = math21_device_f_divide;
        } else if (fname == m21_fname_xk_divide) {
            f = math21_device_f_xk_divide;
        } else if (fname == m21_fname_kx_pow) {
            f = math21_device_f_kx_pow;
        } else if (fname == m21_fname_xk_pow) {
            f = math21_device_f_xk_pow;
        } else {
            MATH21_ASSERT(0, "not support calling function with name " << fname);
        }
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_template_vector_f_kx_like_cpu_kernel(f, n, k, x, y, id);
        }
    }
}


