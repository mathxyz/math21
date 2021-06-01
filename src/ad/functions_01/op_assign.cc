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
#include "mat_eye.h"
#include "num_assign.h"
#include "op_assign.h"

namespace math21 {
    namespace ad {
        op_assign::op_assign() {
        }

        op_assign::~op_assign() {
        }

        NumN op_assign::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);

            op_assign assign0;
            Function &assign = assign0;
            return assign.evaluate(dy, data);
        }

        NumN op_assign::evaluate(const Set &X, VariableMap &data) {
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

        void op_assign::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            // must keep shape 'cause others depend on it.
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);// todo: maybe use tensor share
            const auto &x_vec = data.at(X(1)).getValue();
            auto &y_vec = data.at(y).getValue();
            y_vec = x_vec;
        }

        void op_assign::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            if (X(1) == x) {
                op_mat_eye matEye;
                matEye.f(X, output, data);
            } else {
                output.clear();
                return;
            }
        }

        void op_assign::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X(1) == x) {
                op_assign assign0;
                Function &assign = assign0;
                NumN y_assign;
                assign.f(dy, y_assign, data);
                output.set(y_assign);

//                output.clear();
//                output.add(dy);

//                Set input;
//                input.add(dy);
//                op_assign mat_assign;
//                mat_assign.f(input, output, data);
//                std::string name = math21_string_concatenate("dx = dy in cr ", getName());
//                data.at(output(1)).setName(name.c_str());
            } else {
                output.clear();
//                op_num_constant numConstant;
//                numConstant.f(X, output, data);
                return;
            }
        }

        void op_assign::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X(1) == x) {
                op_assign assign0;
                Function &assign = assign0;
                NumN y_assign;
                assign.forward(dy, y_assign, data);
                output.set(y_assign);
            } else {
                output.clear();
                return;
            }
        }

        void op_assign::f(const Set &X, Set &output, VariableMap &data) {
            MATH21_ASSERT(X.size() == 1)
            NumN y = data.createV("op_assign(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_assign::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            if (Variable::isRequestingAbstractCompletely()) {
                fv(X, Y, data);
                return;
            }
            MATH21_ASSERT (X.size() == 1);

            NumN x = X(1);
            NumN y = Y(1);
            if (data(x).getValue().isScalarInMath()) {
                data.at(y).setAbstractZero();
                auto &y_mat = data.at(y).getVariableMat();
                y_mat.setSize(1);

                op_num_assign num_assign0;
                Function &num_assign = num_assign0;
                NumN y_assign;
                num_assign.f(x, y_assign, data);
                y_mat.at(1) = y_assign;
                std::string name = math21_string_concatenate(getName(),
                                                             math21_string_to_string(1),
                                                             "(x)");
                data.at(y_mat(1)).setName(name.c_str());
            } else {
                const auto &x_mat = data(x).getVariableMat();
                MATH21_ASSERT (!data(x).isAbstractCompletely());

                if (data.at(y).isAbstractCompletely()) {
                    data.at(y).setAbstractZero();
                    auto &y_mat = data.at(y).getVariableMat();
                    y_mat.setSize(x_mat.shape());
                    for (NumN i = 1; i <= y_mat.size(); ++i) {
                        op_num_assign num_assign0;
                        Function &num_assign = num_assign0;
                        NumN y_assign;
                        num_assign.f(x_mat(i), y_assign, data);
                        y_mat.at(i) = y_assign;
                        std::string name = math21_string_concatenate(getName(),
                                                                     math21_string_to_string(i),
                                                                     "(x)");
                        data.at(y_mat(i)).setName(name.c_str());
                    }
                } else {
                    auto &y_mat = data.at(y).getVariableMat();
                    MATH21_ASSERT(y_mat.size() == x_mat.size());
                }
            }
            auto &y_mat = data.at(y).getVariableMat();
            for (NumN i = 1; i <= y_mat.size(); ++i) {
                derivative.compute(y_mat(i));
            }
            data.at(y).synchronizeValue(data);
            data.at(y).setComputed(1);
        }

        void op_assign::setSize(const Set &X, const Set &Y, VariableMap &data) const {
        }

        Function *op_assign::clone() const {
            Function *f = new op_assign();
            return f;
        }

        const char *op_assign::getName() const {
            return "op_assign";
        }
    }
}