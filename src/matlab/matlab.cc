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

#include "01.h"
#include "matlab.h"

namespace math21 {
    namespace matlab {
        // y = k * x
        TenR operator*(NumR k, const TenR &x) {
            TenR y;
            y.setSize(x.shape());
            math21_operator_container_linear_kx_b(k, x, 0, y);
            return y;
        }

        TenR operator+(const NumR &x1, const TenR &x2) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x1, x2, y, xjadd);
            return y;
        }

        TenR operator+(const TenR &x1, const NumR &x2) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x1, x2, y, xjadd);
            return y;
        }

        // It can be matrix + vector.
        TenR operator+(const TenR &x1, const TenR &x2) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x1, x2, y, xjadd);
            return y;
        }

        TenR operator-(const TenR &x1, const TenR &x2) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x1, x2, y, xjsubtract);
            return y;
        }

        TenR operator-(const TenR &x1) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x1, y, xjnegate);
            return y;
        }

        TenR operator*(const TenR &x1, const TenR &x2) {
            TenR y;
            math21_operator_matrix_multiply_general(1, x1, x2, y);
            return y;
        }

        // multiply elementwise, no period character (.) supported
        TenR multiply_elewise(const TenR &x1, const TenR &x2) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x1, x2, y, xjmultiply);
            return y;
        }

        TenR transpose(const TenR &x1) {
            TenR y;
            math21_operator_matrix_trans(x1, y);
            return y;
        }

        TenR inv(const TenR &x1) {
            TenR y;
            math21_operator_inverse(x1, y);
            return y;
        }

        NumR dot(const TenR &x1, const TenR &x2) {
            return math21_operator_container_InnerProduct(1, x1, x2);
        }

        TenR sin(const TenR &x) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x, y, xjsin);
            return y;
        }

        TenR cos(const TenR &x) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x, y, xjcos);
            return y;
        }

        TenR asin(const TenR &x) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x, y, xjasin);
            return y;
        }

        TenR acos(const TenR &x) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x, y, xjacos);
            return y;
        }

        TenR mod(const TenR &x, NumR m) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x, m, y, xjmod);
            return y;
        }

        // modulo operation
        // y = mod(x, m), y = x % m
        TenR mod(const TenR &x, const TenR &m) {
            TenR y;
            math21_operator_tensor_f_elementwise_binary(x, m, y, xjmod);
            return y;
        }

        TenR abs(const TenR &x) {
            TenR y;
            math21_operator_tensor_f_elementwise_unary(x, y, xjabs);
            return y;
        }

        TenR rand(NumN d1, NumN d2) {
            RanUniform ranUniform;
            TenR y(d1, d2);
            math21_random_draw(y, ranUniform);
            return y;
        }

        TenR zeros(NumN d1, NumN d2) {
            TenR y(d1, d2);
            y = 0;
            return y;
        }

        TenR eye(NumN d1) {
            TenR y(d1, d1);
            math21_operator_mat_eye(y);
            return y;
        }

        TenR eye(NumN d1, NumN d2) {
            TenR y(d1, d2);
            math21_operator_mat_eye(y);
            return y;
        }

        NumR norm(const MatR &X, NumN p) {
            if (X.nrows() == 1 || X.ncols() == 1) { // norm of vector
                return math21_operator_container_norm(X, p);
            } else { // norm of matrix
                MATH21_ASSERT(0, "matrix norm not support!")
            }
            return 0;
        }

        // the Frobenius norm of matrix X.
        NumR norm(const MatR &X, const std::string &type) {
            MATH21_ASSERT(type == "fro")
            return math21_operator_container_norm(X, 2);
        }

        MatR operator^(const MatR &A, NumZ n) {
            MatR y;
            math21_matlab_mpower(A, n, y);
            return y;
        }

        MatR diag(const MatR &A, NumZ n) {
            MatR B;
            math21_matlab_diag(A, B, n);
            return B;
        }

        NumR trace(const MatR &A) {
            return math21_operator_matrix_trace(A);
        }
    }
}