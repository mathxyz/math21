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
#include "num_multiply.h"
#include "op_multiply.h"
#include "op_ele_unary.h"

namespace math21 {
    namespace ad {
        op_ele_unary::op_ele_unary() {
        }

        op_ele_unary::~op_ele_unary() {
        }

        NumN op_ele_unary::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(0);
            return 0;
        }

        NumN op_ele_unary::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN dydx = df_vjp(X, x, y, data);
            op_multiply multiply0;
            Function &multiply = multiply0;
            return multiply.evaluate(dy, dydx, data);
        }

        NumN op_ele_unary::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
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

        void op_ele_unary::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            NumN dydx;
            df(X, x, y, dydx, data);

            op_num_multiply multiply0;
            Function &multiply = multiply0;
            NumN dx;
            multiply.f(dy, dydx, dx, data);
            auto name = math21_string_concatenate("dx = dy * d(", getName(), "(x))");
            data.at(dx).setName(name.c_str());
            output.set(dx);
        }

        void op_ele_unary::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            NumN dydx;
            df_dbr(X, x, y, dydx, data);

            op_num_multiply multiply0;
            Function &multiply = multiply0;
            NumN dx;
            multiply.forward(dy, dydx, dx, data);
            auto name = math21_string_concatenate("dx = dy * d(", getName(), "(x))");
            data.at(dx).setName(name.c_str());
            output.set(dx);
        }

        void op_ele_unary::f(const Set &X, Set &Y, VariableMap &data) {
            NumN x = X(1);
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setf(this);
            data.at(y).getValue().setSize(1);
            data.at(y).addx(x);
            data.at(x).addy(y);
            Y.set(y);
        }

        void op_ele_unary::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            // must keep shape 'cause others depend on it.
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);
            const auto &x_vec = data.at(X(1)).getValue();
            auto &y_vec = data.at(y).getValue();
            evaluate_at_tensor(x_vec, y_vec);
//            NumN n = y_vec.size();
//            for (NumN i = 1; i <= n; ++i) {
//                y_vec.at(i) = evaluate_at_num(x_vec(i));
//            }
        }
    }
}