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
#include "op_list_to_i_var.h"
#include "op_vars_to_list.h"

namespace math21 {
    namespace ad {

        op_vars_to_list::op_vars_to_list() {
        }

        op_vars_to_list::~op_vars_to_list() {
        }

        // dy is var list
        NumN op_vars_to_list::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            const auto &dy_value = data(dy).getValue();
            math21_tool_assert(dy_value.size() == X.size());
            op_list_to_i_var _op_list_to_i_var(pos);
            Function &f = _op_list_to_i_var;
            return f.evaluate(dy, data);
        }

        // vars => list => pack (such as row stack)
        // dvars <= dlist <= dpack (such as row stack)
        // ddvars => ddlist => ddpack (such as row stack)
        // x1, x2, x3 -> [x1, x2, x3]
        NumN op_vars_to_list::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() >= 1);
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setType(variable_type_list);
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

        void op_vars_to_list::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            NumN n = X.size();
            auto &y_vec = data.at(y).getValue();
            y_vec.setSize(n);
            for (NumN i = 1; i <= n; ++i) {
                y_vec(i) = X(i);
            }
        }

        void op_vars_to_list::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vars_to_list::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vars_to_list::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_vars_to_list::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_vars_to_list::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_vars_to_list::clone() const {
            Function *f = new op_vars_to_list();
            return f;
        }

        const char *op_vars_to_list::getName() const {
            return "op_vars_to_list";
        }
    }
}