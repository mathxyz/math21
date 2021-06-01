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
#include "op_merge_vectors.h"

namespace math21 {
    namespace ad {

        op_merge_vectors::op_merge_vectors() {
        }

        op_merge_vectors::~op_merge_vectors() {
        }

        NumN op_merge_vectors::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(0);
            return 0;
        }

        // x1, x2 are seen as vectors if they are not.
        // x1, x2 => y, y = (x1, x2), here x1, x2, y are all vectors.
        NumN op_merge_vectors::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() == 2);
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

        void op_merge_vectors::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            const auto &x1_value = data.at(X(1)).getValue();
            const auto &x2_value = data.at(X(2)).getValue();
            auto &y_value = data.at(y).getValue();
//            math21_operator_merge(x1_value, x2_value, y_value);
            math21_op_vector_concatenate(x1_value, x2_value, y_value);
        }

        void op_merge_vectors::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_merge_vectors::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_merge_vectors::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_merge_vectors::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_merge_vectors::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_merge_vectors::clone() const {
            Function *f = new op_merge_vectors();
            return f;
        }

        const char *op_merge_vectors::getName() const {
            return "op_merge_vectors";
        }
    }
}