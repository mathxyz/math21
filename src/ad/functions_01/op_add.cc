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
#include "op_assign.h"
#include "num_add.h"
#include "mat_eye.h"
#include "op_add.h"

namespace math21 {
    namespace ad {
        op_add::op_add() {
        }

        op_add::~op_add() {
        }

        NumN op_add::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() >= 1);
            math21_tool_assert(X.contains(x));

            op_assign assign0;
            Function &assign = assign0;
            return assign.evaluate(dy, data);
        }

        NumN op_add::evaluate(const Set &X0, VariableMap &data) {
            math21_tool_assert(X0.size() >= 1);
            Set X;
            broadcast_num_to_vec(X0, X, data);
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

        void op_add::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            // must keep shape 'cause others depend on it.
            variable_setSize_to_same_vspace_using_variable(X(1), y, data);
            auto &y_vec = data.at(y).getValue();
            y_vec = 0;
            for (NumN i = 1; i <= X.size(); ++i) {
                math21_op_container_addToB(data(X(i)).getValue(), y_vec);
            }
        }

        NumN op_add::evaluate_inc(const Set &X0, NumN y, VariableMap &data) {
            if(y==0){
                return evaluate(X0, data);
            }
            math21_tool_assert(X0.size() >= 1);
            Set X;
            Set X0_y;
            Set X_y;
            X0_y.add(y);
            X0_y.add(X0);
            broadcast_num_to_vec(X0_y, X_y, data);
            math21_tool_assert(X_y(1)==y);
            X_y.copyToNox(X, y);
            data.at(y).addX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            auto &y_vec = data.at(y).getValue();
            for (NumN i = 1; i <= X.size(); ++i) {
                math21_op_container_addToB(data(X(i)).getValue(), y_vec);
            }
            return y;
        }

        void op_add::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            if (X.contains(x)) {
                op_mat_eye matEye;
                matEye.f(X, output, data);
            } else {
                output.clear();
                return;
            }
        }

        void op_add::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X.contains(x)) {
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
            }
        }

        void op_add::f(const Set &X, Set &output, VariableMap &data) {
            if (X.size() == 0) {
                output.clear();
                return;
            }
            if (X.size() == 1) {
                op_assign assign0;
                Function &assign = assign0;
                assign.f(X, output, data);
//                output.set(X(1)); // todo: has some error
                return;
            }
            NumN y = data.createV("op_add(x)");
            if (isSetSize()) {
                setSizeyByx(X(1), y, data);
            }
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_add::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            if (Variable::isRequestingAbstractCompletely()) {
                fv(X, Y, data);
                return;
            }
            if (X.size() == 0) {
                return;
            }
            if (X.size() == 1) {
                op_assign assign0;
                Function &assign = assign0;
                assign.compute(X, Y, data, derivative);
                return;
            }

            NumN x = X(1);
            NumN y = Y(1);
            if (data(x).getValue().isScalarInMath()) {
                if (data.at(y).isAbstractCompletely()) {
                    data.at(y).setAbstractZero();
                    auto &y_mat = data.at(y).getVariableMat();
                    y_mat.setSize(1);

                    op_num_add num_add0;
                    Set Y_add;
                    Function &num_add = num_add0;
                    num_add.f(X, Y_add, data);
                    y_mat.at(1) = Y_add(1);
                    std::string name = math21_string_concatenate(getName(),
                                                                 math21_string_to_string(1),
                                                                 "(x)");
                    data.at(y_mat(1)).setName(name.c_str());

                }
            } else {
                const auto &x_mat = data(x).getVariableMat();
                MATH21_ASSERT (!data(x).isAbstractCompletely());

                if (data.at(y).isAbstractCompletely()) {
                    data.at(y).setAbstractZero();
                    auto &y_mat = data.at(y).getVariableMat();
                    y_mat.setSize(x_mat.shape());
                    for (NumN i = 1; i <= y_mat.size(); ++i) {
                        op_num_add num_add0;
                        Set X_add;
                        Set Y_add;
                        for (NumN j = 1; j <= X.size(); ++j) {
                            NumN xj = X(j);
                            const auto &xj_mat = data(xj).getVariableMat();
                            X_add.add(xj_mat(i));
                        }
                        Function &num_add = num_add0;
                        num_add.f(X_add, Y_add, data);
                        y_mat.at(i) = Y_add(1);
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

        void op_add::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            setSizeCXUXByX(X, data);
            setSizeYByX(X, Y, data);
        }

        Function *op_add::clone() const {
            Function *f = new op_add();
            return f;
        }

        const char *op_add::getName() const {
            return "op_add";
        }
    }
}