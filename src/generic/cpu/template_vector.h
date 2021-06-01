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

namespace math21 {

    template<typename T>
    NumB math21_template_vector_is_equal_cpu(NumN n, const T *x, const T *y, NumR epsilon, NumN logLevel) {
        x -= 1;
        y -= 1;
        NumN id;
//#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            NumR tmp = (NumR) y[id] - (NumR) x[id];
            if (xjabs(tmp) > epsilon) {
                break;
            }
        }
        if (id <= n) {
            if (logLevel) {
                printf("different from postion %d\n", id);
            }
            return 0;
        }
        return 1;

    }

    template<typename T>
    NumR math21_template_vector_distance(NumN n, const T *A, const T *B, NumR norm) {
        MATH21_ASSERT(norm > 0);
        A -= 1;
        B -= 1;
        NumN i;
        NumR sum = 0;
        if (norm == 1) {
//#pragma omp parallel for
            for (i = 1; i <= n; ++i) sum += xjabs(A[i] - B[i]);
        } else if (norm == 2) {
//#pragma omp parallel for
            for (i = 1; i <= n; ++i) sum += xjsquare(A[i] - B[i]);
            sum = xjsqrt(sum);
        } else {
            for (i = 1; i <= n; ++i) sum += xjpow(xjabs(A[i] - B[i]), norm);
            sum = xjpow(sum, 1 / norm);
        }
        MATH21_ASSERT_FINITE(math21_operator_isfinite(sum))
        return sum;
    }

    template<typename T>
    T math21_template_vector_max(NumN n, const T *x, NumN &index) {
        x -= 1;
        NumN i;
        MATH21_ASSERT(n >= 1);
        NumN k = 1;
        for (i = 1; i <= n; ++i) {
            if (x[i] > x[k]) {
                k = i;
            }
        }
        index = k;
        return x[k];
    }

    template<typename T>
    T math21_template_vector_min(NumN n, const T *x, NumN &index) {
        x -= 1;
        NumN i;
        MATH21_ASSERT(n >= 1);
        NumN k = 1;
        for (i = 1; i <= n; ++i) {
            if (x[i] < x[k]) {
                k = i;
            }
        }
        index = k;
        return x[k];
    }

    // see math21_operator_matrix_reverse_y_axis
    template<typename T>
    void math21_template_tensor_reverse_axis_3_in_d3_cpu(T *x, NumN d1, NumN d2, NumN d3) {
        x -= 1;
        NumN n = d1 * d2 * (d3 / 2);
        NumN id;
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            NumN i1, i2, i3, ix, iy;
            math21_device_index_1d_to_3d_fast(&i1, &i2, &i3, id, d2, d3 / 2);
            math21_device_index_3d_to_1d_fast(i1, i2, i3, &ix, d2, d3);
            math21_device_index_3d_to_1d_fast(i1, i2, d3 + 1 - i3, &iy, d2, d3);
            m21_swap(x[ix], x[iy]);
        }
    }

}