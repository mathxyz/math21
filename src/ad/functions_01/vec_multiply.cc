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

#include "vec_multiply.h"

namespace math21 {
    namespace ad {
        op_vec_multiply::op_vec_multiply() {
        }

        op_vec_multiply::~op_vec_multiply() {
        }

        void op_vec_multiply::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            Set X1;
            X.copyToNox(X1, x);

            op_vec_multiply multiply;
            if (X1.isEmpty()) {
                NumN dx = data.createC("1=d(op_vec_multiply(x))");
                data.setValue(dx, 1);
                if (isSetSize()) {
                    setSizeyByx(x, dx, data);
                }
                output.clear();
                output.add(dx);
            } else {
                multiply.f(X1, output, data);
                data.at(output(1)).setName("d(op_vec_multiply(x))");
            }

            Set input;
            input.add(dy);
            input.add(output);
            multiply.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_vec_multiply(x))");
        }

        void op_vec_multiply::f(const Set &X, Set &output, VariableMap &data) {
            NumN y = data.createV("op_vec_multiply(x)");
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

        // todo: k*x*y
        // dot product
        void op_vec_multiply::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            if(data.at(y).isComputed()){
                return;
            }
            data.at(y).getValue().assign(data.at(X(1)).getValue());
            for (NumN i = 2; i <= X.size(); ++i) {
                math21_operator_SchurProduct_to_B(data(X(i)).getValue(), data.at(y).getValue());
            }
            data.at(y).setComputed(1);
        }

        void op_vec_multiply::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            setSizeCXUXByX(X, data);
            setSizeYByX(X, Y, data);
        }


        Function *op_vec_multiply::clone() const {
            Function *f = new op_vec_multiply();
            return f;
        }

        const char *op_vec_multiply::getName() const {
            return "op_vec_multiply";
        }
    }
}