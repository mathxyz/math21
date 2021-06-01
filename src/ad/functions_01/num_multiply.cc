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

#include "num_constant.h"
#include "inner_cc.h"
#include "num_multiply.h"

namespace math21 {
    namespace ad {
        op_num_multiply::op_num_multiply() {
        }

        op_num_multiply::~op_num_multiply() {
        }

        void op_num_multiply::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void op_num_multiply::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (!X.contains(x)) {
                output.set(ad_global_get_constant_0());
                return;
            }
            if (X.size() == 1) {
                output.set(ad_global_get_constant_1());
            } else {
                Set X1;
                X.copyToNox(X1, x);
                op_num_multiply multiply;
                multiply.f(X1, output, data);
                data.at(output(1)).setName("d(op_num_multiply(x))");
            }

            Set input;
            input.add(dy);
            input.add(output);
            op_num_multiply multiply;
            multiply.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_num_multiply(x))");
        }

        void op_num_multiply::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (!X.contains(x)) {
                output.set(ad_global_get_constant_0());
                return;
            }
            if (X.size() == 1) {
                output.set(ad_global_get_constant_1());
            } else {
                Set X1;
                X.copyToNox(X1, x);
                op_num_multiply multiply;
                multiply.forward(X1, output, data);
                data.at(output(1)).setName("d(op_num_multiply(x))");
            }

            Set input;
            input.add(dy);
            input.add(output);
            op_num_multiply multiply;
            multiply.forward(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_num_multiply(x))");
        }

        void op_num_multiply::f(const Set &X, Set &output, VariableMap &data) {
            if (ad_is_containing_constant_num_0(X, data)) {
                output.set(ad_global_get_constant_0());
                return;
            }
            NumN y = data.createV("op_num_multiply(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_num_multiply::fv(const Set &X, const Set &Y, VariableMap &data) const {
            if (X.isEmpty()) {
                return;
            }
            NumN y = Y(1);
            // this checking has its corresponding part in f.
            // It is used when calling f and fv sequentially.
            if(ad_is_constant_num(y, data)){
                return;
            }
            if (data.at(y).isComputed()) {
                return;
            }
            data.at(y).getValue().setSize(1);
            NumR sum = 1;
            for (NumN i = 1; i <= X.size(); ++i) {
                sum = sum * data(X(i)).getValue()(1);
            }
            data.at(y).getValue() = sum;
            data.at(y).setComputed(1);
        }

        void op_num_multiply::forward(const Set &X, Set &output, VariableMap &data) {
            if (X.isEmpty()) {
                return;
            }
            if (ad_is_containing_constant_num_0(X, data)) {
                output.set(ad_global_get_constant_0());
                return;
            }
            NumN y = data.createV("op_num_multiply(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);

            data.at(y).getValue().setSize(1);
            NumR sum = 1;
            for (NumN i = 1; i <= X.size(); ++i) {
                sum = sum * data(X(i)).getValue()(1);
            }
            data.at(y).getValue() = sum;
            data.at(y).setComputed(1);
        }

        void op_num_multiply::setSize(const Set &X, const Set &Y, VariableMap &data) const {
        }

        Function *op_num_multiply::clone() const {
            Function *f = new op_num_multiply();
            return f;
        }

        const char *op_num_multiply::getName() const {
            return "op_num_multiply";
        }
    }
}