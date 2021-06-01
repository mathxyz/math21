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

#include "matlab.h"
#include "01.h"

namespace math21 {
    using namespace matlab;

    // todo: check
    void math21_operator_container_create_with_increment(VecR &v, NumR from, NumR inc, NumR to, NumR epsilon, NumN n) {
        MATH21_ASSERT(inc != 0);
        if (n == 0) {
            NumZ n0 = std::floor((to - from) / inc + epsilon) + 1;
            MATH21_ASSERT(n0 > 0);
            n = n0;
        }
        v.setSize(n);
        NumN i;
        for (i = 1; i <= n; ++i) {
            v(i) = from + (i - 1) * inc;
        }
    }

    void math21_matlab_vector_create_with_increment(VecR &v, NumR from, NumR inc, NumR to, NumR epsilon) {
        math21_operator_container_create_with_increment(v, from, inc, to, epsilon, 0);
    }

    void math21_matlab_vector_create_with_unit_increment(VecR &v, NumR from, NumR to) {
        math21_matlab_vector_create_with_increment(v, from, 1, to);
    }

    void math21_matlab_mpower(const MatR &A, NumZ n, MatR &y) {
        if (n == -1) {
            math21_operator_inverse(A, y);
        } else if (n > 0) {
            NumN i;
            y = A;
            MatR res;
            for (i = 2; i <= (NumN) n; ++i) {
                math21_operator_matrix_multiply_general(1, y, A, res);
                y.swap(res);
            }
        } else {
            MATH21_ASSERT(0, "not support!")
        }
    }

    void math21_matlab_diag(const MatR &A, MatR &B, NumZ n) {
        if (n == 0) {
            if (A.isVectorInMath()) {
                B.setSize(A.size(), A.size());
                B = 0;
                math21_operator_matrix_diagonal_set(B, A);
            } else {
                math21_operator_matrix_diagonal_get(A, B);
            }
        } else {
            MATH21_ASSERT(0, "not support!")
        }
    }

    /*
# References
        - [A compact formula for the derivative of
        a 3-D rotation in exponential coordinates](https://arxiv.org/abs/1312.0788)
 * */
    void math21_la_convert_RodriguesForm_to_rotation(const VecR &w, MatR &R) {
        if (!R.isSameSize(3, 3)) {
            R.setSize(3, 3);
        }
        NumR wx, wy, wz;
        wx = w(1);
        wy = w(2);
        wz = w(3);
        MatR omega(3, 3);
        omega =
                0, -wz, wy,
                wz, 0, -wx,
                -wy, wx, 0;
        NumR theta = norm(w);
        R = eye(3) + (std::sin(theta) / theta) * omega + ((1 - std::cos(theta)) / std::pow(theta, 2)) * (omega ^ 2);
    }

    void math21_la_convert_rotation_to_RodriguesForm(const MatR &R, VecR &w) {
        if (!w.isSameSize(3)) {
            w.setSize(3);
        }
        NumR theta = std::acos((trace(R) - 1) / 2.0);
        w = R(3, 2) - R(2, 3), R(1, 3) - R(3, 1), R(2, 1) - R(1, 2);
        w = (theta / (2 * std::sin(theta))) * w;
    }
}