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
#include "op_multiply.h"
#include "op_sin.h"

namespace math21 {
    namespace ad {
        NumN op_sin::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            op_cos cos0;
            Function &cos = cos0;
            NumN y_cos = cos.evaluate(x, data);
            op_multiply multiply0;
            Function &multiply = multiply0;
            return multiply.evaluate(dy, y_cos, data);
        }

        NumN op_sin::evaluate(const Set &X, VariableMap &data) {
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

        void op_sin::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);
            const auto &x_vec = data.at(X(1)).getValue();
            auto &y_vec = data.at(y).getValue();
            math21_op_container_sin(x_vec, y_vec);
        }

        NumN op_cos::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);

            NumN k = data.createC("-1");
            variable_set_device_type_using_variable(x, k, data);
            MATH21_ASSERT(data.at(k).getValue().isEmpty(), "just check")
//            data.setValue(k, -1);
            data.at(k).getValue() = -1;
            MATH21_ASSERT(0, "just check")
            MATH21_ASSERT(!data.at(k).getValue().isEmpty(), "just check")

            op_sin sin0;
            Function &sin = sin0;
            NumN y_sin = sin.evaluate(x, data);
            op_multiply multiply0;
            Function &multiply = multiply0;
            return multiply.evaluate(k, dy, y_sin, data);
        }

        NumN op_cos::evaluate(const Set &X, VariableMap &data) {
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

        void op_cos::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);
            const auto &x_vec = data.at(X(1)).getValue();
            auto &y_vec = data.at(y).getValue();
            math21_op_container_cos(x_vec, y_vec);
        }
    }
}