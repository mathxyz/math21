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
#include "op_vec_share_sub.h"

namespace math21 {
    namespace ad {

        op_vec_share_sub::op_vec_share_sub(NumZ from, NumZ to) : from(from), to(to) {
        }

        op_vec_share_sub::~op_vec_share_sub() {
        }

        // dy is var list
        NumN op_vec_share_sub::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            const auto &x_value = data.at(x).getValue();
            op_vec_d_sub _op_vec_d_sub(x_value.size(), from, to);
            Function &f = _op_vec_d_sub;
            return f.evaluate(dy, data);
        }

        // submatrix operation for vector
        NumN op_vec_share_sub::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() == 1);
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

        void op_vec_share_sub::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            NumN x = X(1);
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            math21_operator_share_vector_part_using_from_to(x_value, y_value, from, to);
        }

        void op_vec_share_sub::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vec_share_sub::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vec_share_sub::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_vec_share_sub::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_vec_share_sub::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_vec_share_sub::clone() const {
            Function *f = new op_vec_share_sub(from, to);
            return f;
        }

        const char *op_vec_share_sub::getName() const {
            return "op_vec_share_sub";
        }
    }

    namespace ad {

        op_vec_d_sub::op_vec_d_sub(NumN n, NumZ from, NumZ to) : n(n), from(from), to(to) {
        }

        op_vec_d_sub::~op_vec_d_sub() {
        }

        NumN op_vec_d_sub::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            op_vec_share_sub _op_vec_share_sub(from, to);
            Function &f = _op_vec_share_sub;
            NumN dx = f.evaluate(dy, data);
            // If op_assign is omitted, dx will share part data of dy
            NumB useShared = 1;
//            NumB useShared = 0;
            if (!useShared) {
                op_assign assign0;
                Function &assign = assign0;
                dx = assign.evaluate(dx, data);
            }
            return dx;
        }

        // vjp of submatrix operation for vector
        NumN op_vec_d_sub::evaluate(const Set &X, VariableMap &data) {
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

        void op_vec_d_sub::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            NumN x = X(1);
            const auto &x_value = data.at(x).getValue();
            auto &y_value = data.at(y).getValue();
            y_value.setSize(n);
            y_value = 0;
//            VecR y_part_value;
//            math21_operator_share_vector_part_using_from_to(y_value, y_part_value, from, to);
//            math21_operator_container_set(x_value, y_part_value);

            NumN _from = math21_number_container_pos_check(n, from);
            NumN offset = _from - 1;
            math21_op_vector_sub_region_set(x_value, y_value, 0, offset, 0);
        }

        void op_vec_d_sub::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vec_d_sub::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            math21_tool_assert(0);
        }

        void op_vec_d_sub::f(const Set &X, Set &output, VariableMap &data) {
            math21_tool_assert(0);
        }

        void op_vec_d_sub::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            math21_tool_assert(0);
        }

        void op_vec_d_sub::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(0);
        }

        Function *op_vec_d_sub::clone() const {
            Function *f = new op_vec_d_sub(n, from, to);
            return f;
        }

        const char *op_vec_d_sub::getName() const {
            return "op_vec_d_sub";
        }
    }
}