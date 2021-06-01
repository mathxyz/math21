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
#include "op_get_shape.h"
#include "op_share_reshape.h"

namespace math21 {
    namespace ad {

        op_share_reshape::op_share_reshape() {
        }

        op_share_reshape::~op_share_reshape() {
        }

        NumN op_share_reshape::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            MATH21_ASSERT(X(1) == x);
            const auto &x_value = data.at(x).getValue();
            op_get_shape _get_shape;
            Function &f_get_shape = _get_shape;
            NumN d = f_get_shape.evaluate(x, data);
            op_share_reshape _op_share_reshape;
            Function &f = _op_share_reshape;
            return f.evaluate(dy, d, data);
        }

        // no need to set device type for y
        NumN op_share_reshape::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() == 2);
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

        void op_share_reshape::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            const auto &x_value = data.at(X(1)).getValue();
            const auto &d_value = data.at(X(2)).getValue();
            MATH21_ASSERT(d_value.dims() == 1)
            VecN d;
            d = d_value;
            auto &y_value = data.at(y).getValue();
            math21_operator_tensor_shallow_copy(x_value, y_value);
            variable_reshape_to_same_vspace_using_shape(d, y, data);
        }

        void op_share_reshape::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_share_reshape::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_share_reshape::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_share_reshape::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_share_reshape::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_share_reshape::clone() const {
            Function *f = new op_share_reshape();
            return f;
        }

        const char *op_share_reshape::getName() const {
            return "op_share_reshape";
        }
    }
}