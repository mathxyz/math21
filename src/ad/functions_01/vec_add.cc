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
#include "vec_multiply.h"
#include "vec_add.h"

namespace math21 {
    namespace ad {
        op_vec_add::op_vec_add() {
        }

        op_vec_add::~op_vec_add() {
        }

        void op_vec_add::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            NumN dx = data.createC("1=d(op_vec_add(x))");
            data.setValue(dx, 1);
            if (isSetSize()) {
                setSizeyByx(x, dx, data);
            }
            Set input;
            input.add(dy);
            input.add(dx);
            op_vec_multiply multiply;
            multiply.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_vec_add(x))");
        }

        void op_vec_add::f(const Set &X, Set &output, VariableMap &data) {
            if (X.size() == 0) {
                output.clear();
                return;
            }
            if (X.size() == 1) {
                output.clear();
                output.add(X(1));
                return;
            }
            NumN y = data.createV("op_vec_add(x)");
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

        void op_vec_add::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN x1 = X(1);
            NumN y = Y(1);
            if(data.at(y).isComputed()){
                return;
            }
            const auto &x1_mat = data(x1).getValue();
            auto &y_mat = data.at(y).getValue();
            if (y_mat.size() != x1_mat.size()) {
                y_mat.setSize(x1_mat.shape());
            }
            y_mat = 0;
            for (NumN i = 1; i <= X.size(); ++i) {
                math21_operator_container_addToB(data(X(i)).getValue(), data.at(y).getValue());
            }
            data.at(y).setComputed(1);
        }

        void op_vec_add::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            setSizeCXUXByX(X, data);
            setSizeYByX(X, Y, data);
        }

        Function *op_vec_add::clone() const {
            Function *f = new op_vec_add();
            return f;
        }

        const char *op_vec_add::getName() const {
            return "op_vec_add";
        }
    }
}