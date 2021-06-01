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
#include "op_m_to_n.h"

namespace math21 {
    namespace ad {
        op_m_to_n::op_m_to_n() {
        }

        op_m_to_n::~op_m_to_n() {
        }

        NumN op_m_to_n::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(0);
            return 0;
        }

        NumN op_m_to_n::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN dydx = df_vjp(X, x, y, data);
            NumN dx;
            // optimized for dy size 1, use scalar mul instead of mat mul to speed up.
            if (data(dy).getValue().size() == 1) {
                op_multiply multiply0;
                Function &multiply = multiply0;
                dx = multiply.evaluate(dy, dydx, data);
            } else {
                op_mat_mul multiply0;
                Function &multiply = multiply0;
                dx = multiply.evaluate(dy, dydx, data);
            }
            return dx;
        }

        NumN op_m_to_n::evaluate(const Set &X, VariableMap &data) {
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

        void op_m_to_n::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            auto &y_vec = data.at(y).getValue();
            NumN n_var = X.size();
            if (n_var == 1) {
                const auto &x_vec = data.at(X(1)).getValue();
                evaluate_at_vec(x_vec, y_vec);
            } else if (n_var == 2) {
                const auto &x1_vec = data.at(X(1)).getValue();
                const auto &x2_vec = data.at(X(2)).getValue();
                evaluate_at_vec(x1_vec, x2_vec, y_vec);
            } else if (n_var == 3) {
                const auto &x1_vec = data.at(X(1)).getValue();
                const auto &x2_vec = data.at(X(2)).getValue();
                const auto &x3_vec = data.at(X(3)).getValue();
                evaluate_at_vec(x1_vec, x2_vec, x3_vec, y_vec);
            } else {
                MATH21_ASSERT(0)
            }
        }

        void op_m_to_n::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_m_to_n::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_m_to_n::f(const Set &X, Set &Y, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_m_to_n::evaluate_at_vec(const VecR &x1, VecR &y) const {
            math21_tool_assert(0 && "Please overwrite to use!");
        }

        void op_m_to_n::evaluate_at_vec(const VecR &x1, const VecR &x2, VecR &y) const {
            math21_tool_assert(0 && "Please overwrite to use!");
        }

        void op_m_to_n::evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const {
            math21_tool_assert(0 && "Please overwrite to use!");
        }
    }
}