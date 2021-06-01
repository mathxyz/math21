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
#include "op_push.h"

namespace math21 {
    namespace ad {

        NumN op_push::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            op_pull _op_pull;
            Function &f_op_pull = _op_pull;
            return f_op_pull.evaluate(dy, data);
        }

        NumN op_push::evaluate(const Set &X, VariableMap &data) {
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            variable_set_device_type_gpu(y, data);
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

        void op_push::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            const auto &x_value = data(X(1)).getValue();
            math21_tool_assert(x_value.is_cpu());
            y_value = x_value;
        }

        NumN op_pull::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            op_push _op_push;
            Function &f_op_push = _op_push;
            return f_op_push.evaluate(dy, data);
        }

        NumN op_pull::evaluate(const Set &X, VariableMap &data) {
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
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

        void op_pull::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            const auto &x_value = data(X(1)).getValue();
            math21_tool_assert(!x_value.is_cpu());
            y_value = x_value;
        }
    }
}