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
#include "op_mat_trans.h"

namespace math21 {
    namespace ad {
        op_mat_trans::op_mat_trans() {
        }

        op_mat_trans::~op_mat_trans() {
        }

        NumN op_mat_trans::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            const auto &y_mat = data.at(y).getValue();
            auto &dy_mat = data.at(dy).getValue();
            MATH21_ASSERT(dy_mat.isSameSize(y_mat.shape()), "Design check! See Function::cr_vjp()")
//            dy_mat.reshape(y_mat.shape());

            op_mat_trans _op_mat_trans;
            Function &f = _op_mat_trans;
            return f.evaluate(dy, data);
        }

        NumN op_mat_trans::evaluate(const Set &X, VariableMap &data) {
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

        void op_mat_trans::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            const auto &x_mat = data.at(X(1)).getValue();
            auto &y_mat = data.at(y).getValue();
//            math21_operator_matrix_trans(x_mat, y_mat);
            math21_op_matrix_trans(x_mat, y_mat);
        }

        void op_mat_trans::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented")
        }

        void op_mat_trans::f(const Set &X, Set &output, VariableMap &data) {
            if (X.size() == 0) {
                return;
            }
            MATH21_ASSERT(X.size() == 1)
            NumN y = data.createV("mat_trans(x)");
            if (isSetSize()) {
                math21_operator_tensor_matrix_trans_setSize(
                        data.at(X(1)).getValue(),
                        data.at(y).getValue());
            }
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_mat_trans::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented");
        }

        Function *op_mat_trans::clone() const {
            Function *f = new op_mat_trans();
            return f;
        }

        const char *op_mat_trans::getName() const {
            return "mat_trans";
        }
    }
}