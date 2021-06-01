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

#include "inner_cc.h"
#include "op_row_unpack_i.h"
#include "op_row_pack.h"

namespace math21 {
    namespace ad {
        op_row_pack::op_row_pack() {
        }

        op_row_pack::~op_row_pack() {
        }

        NumN op_row_pack::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            op_row_unpack_i _op_row_unpack_i(pos);
            Function &f = _op_row_unpack_i;
            NumN dx = f.evaluate(dy, data);
            return dx;
        }

        // x1, x2, x3 -> [x1, x2, x3]^t
        NumN op_row_pack::evaluate(const Set &X0, VariableMap &data) {
            math21_tool_assert(X0.size() >= 1);
            Set X;
            broadcast_num_to_vec(X0, X, data);
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_using_variable(X(1), y, data);
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            Set Y;
            Y.add(y);
            fv(X, Y, data);
            return y;
        }

        // can use memcpy to speed up.
        void op_row_pack::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            NumN n = X.size();
            const auto &x1_vec = data.at(X(1)).getValue();
            NumN size = x1_vec.size();
            auto &y_mat = data.at(y).getValue();
            {
                VecN d(2);
                d = n, size;
                if (y_mat.size() == n * size) {
                    variable_reshape_to_same_vspace_using_shape(d, y, data);
                } else {
                    variable_setSize_to_same_vspace_using_shape(d, y, data);
                }
            }

            for (NumN i = 1; i <= n; ++i) {
                const auto &xi_vec = data.at(X(i)).getValue();
                math21_tool_assert(xi_vec.size() == size);
                math21_op_matrix_set_row(xi_vec, y_mat, i);
//                math21_operator_matrix_row_set_by_vec(y_mat, i, xi_vec);
            }

            VecN d_n(1);
            d_n = n;
            VecN d;
            math21_operator_merge(d_n, x1_vec.shape(), d);
            variable_reshape_to_same_vspace_using_shape(d, y, data);
        }

        void op_row_pack::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_row_pack::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_row_pack::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_row_pack::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_row_pack::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_row_pack::clone() const {
            Function *f = new op_row_pack();
            return f;
        }

        const char *op_row_pack::getName() const {
            return "op_row_pack";
        }
    }
}