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

#include "../algebra/set.h"
#include "../matrix_op/files.h"
#include "tenBroadcast.h"

namespace math21 {
    NumB math21_broadcast_is_compatible_calculate(const MatN &x, NumN nc, VecN &d) {
        VecN v;
        for (NumN i = 1; i <= nc; ++i) {
            math21_operator_matrix_col_get(x, i, v);
            SetN S;
            S.add(v);
            S.sort();
            if (S.size() > 3) {
                return 0;
            } else if (S.size() == 3) {
                if (S(1) == 0 && S(2) == 1) {
                    d(i) = 0;
                } else {
                    return 0;
                }
            } else if (S.size() == 2) {
                if (S(1) == 1) {
                    d(i) = S(2);
                } else if (S(1) == 0) {
                    d(i) = 0;
                } else {
                    return 0;
                }
            } else {
                d(i) = S(1);
            }
        }
        return 1;
    }

    // Shapes are considered compatible when each dimension in one shape
    // is either exactly equal to the same dimension in the other shape,
    // or is equal to 1 or 0.
    // e.x., shape 1*2*3 and 3*2*1 is compatible when broadcasting.
    // 3 and 3*2 compatible => d = 3*2
    // 3 and 1*2 compatible => d = 3*2
    // 0 and 1*2 compatible => d = 0*2
    // 3 and 2*1 not compatible
    // 0*1, 1*1, 3*2 compatible => d = 0*2
    // 0*1, 2*1, 3*2 not compatible
    NumB math21_broadcast_is_compatible_in_ele_op(
            const Seqce<VecN> &shapes, VecN &d) {
        MatN x;
        math21_operator_container2d_to_matrix(shapes, x, 1);
        if (x.isEmpty()) {
            return 0;
        }
        VecN v;
        d.setSize(x.ncols());
        return math21_broadcast_is_compatible_calculate(x, x.ncols(), d);
    }

    // shape A, shape B => shape C in stacked matmul
    // C = A*B, * is stacked matmul
    NumB math21_broadcast_is_compatible_in_stacked_matmul(
            const VecN &d_A, const VecN &d_B,
            VecN &d_A_new, VecN &d_B_new, VecN &d_C,
            VecN &d_A_standard, VecN &d_B_standard) {
        Seqce<VecN> shapes(2);
        shapes.at(1) = d_A;
        shapes.at(2) = d_B;
        MatN x;
        math21_operator_container2d_to_matrix(shapes, x, 1);
        if (x.isEmpty()) {
            return 0;
        }
        VecN v;
        NumN n = x.ncols();
        MATH21_ASSERT(n > 1)
        d_C.setSize(n);
        if (n > 2) {
            if (!math21_broadcast_is_compatible_calculate(x, n - 2, d_C)) {
                return 0;
            }
        }
        if (x(1, n) != x(2, n - 1)) {
            return 0;
        }
        d_C(n - 1) = x(1, n - 1);
        d_C(n) = x(2, n);
        d_A_new = d_C;
        d_B_new = d_C;
        d_A_new(n - 1) = x(1, n - 1);
        d_A_new(n) = x(1, n);
        d_B_new(n - 1) = x(2, n - 1);
        d_B_new(n) = x(2, n);
        math21_operator_matrix_row_get(x, 1, d_A_standard);
        math21_operator_matrix_row_get(x, 2, d_B_standard);
        return 1;
    }
}

