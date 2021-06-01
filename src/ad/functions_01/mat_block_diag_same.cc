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

#include "mat_block_diag_same.h"

namespace math21 {
    namespace ad {
        Function_mat_block_diag_same::Function_mat_block_diag_same() {
        }

        Function_mat_block_diag_same::~Function_mat_block_diag_same() {
        }

        void Function_mat_block_diag_same::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X.size() == 0) {
                return;
            }
            MATH21_ASSERT(X.size() <= 2, "not implemented")
            Set X1;
            X.copyToNox(X1, x);
            Set input;
            Function_mat_block_diag_same mat_block_diag;
            if (X1.isEmpty()) {
                // ..
//                NumN dx = data.createC(1, "1=d(mat_block_diag_same(x))");
//                if (isSetSize()) {
//                    setSizeyByx(x, dx, data);
//                }
//                output.clear();
//                output.add(dx);
            } else {
//                mat_block_diag.f(X1, output, data);
//                data.at(output(1)).setName("d(mat_block_diag_same(x))");
            }

            input.clear();
            input.add(dy);
            input.add(output);
            mat_block_diag.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(mat_block_diag_same(x))");
        }

        void Function_mat_block_diag_same::f(const Set &X, Set &output, VariableMap &data) {
            if (X.size() == 0) {
                return;
            }
            if (X.size() == 1) {
                output.clear();
                output.add(X(1));
                return;
            }
            MATH21_ASSERT(X.size() == 2, "not implemented")
            NumN y = data.createV("mat_block_diag_same(x)");
            if (isSetSize()) {
                math21_operator_tensor_matrix_multiply_setSize(
                        data.at(X(1)).getValue(),
                        data.at(X(2)).getValue(),
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

        // dot product
        void Function_mat_block_diag_same::fv(const Set &X, const Set &Y, VariableMap &data) const {
            if (X.size() == 0) {
                return;
            }
            NumN y = Y(1);
            if (X.size() == 1) {
                NumN x = X(1);
                if (x != y) {
                    data.at(y).getValue().assign(data.at(X(1)).getValue());
                }
                return;
            }
            MATH21_ASSERT(X.size() == 2, "not implemented")
            math21_operator_tensor_matrix_multiply_no_setSize(
                    1,
                    data(X(1)).getValue(),
                    data(X(2)).getValue(),
                    data.at(y).getValue()
            );
            data.at(Y(1)).setComputed(1);
        }

        void Function_mat_block_diag_same::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented");
        }

        Function *Function_mat_block_diag_same::clone() const {
            Function *f = new Function_mat_block_diag_same();
            return f;
        }

        const char *Function_mat_block_diag_same::getName() const {
            return "mat_block_diag_same";
        }
    }
}