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

#include "num_multiply.h"
#include "num_constant.h"
#include "inner_cc.h"
#include "num_assign.h"

namespace math21 {
    namespace ad {
        op_num_assign::op_num_assign() {
        }

        op_num_assign::~op_num_assign() {
        }

        void op_num_assign::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            if (X(1) == x) {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                data.at(output(1)).getValue() = 1;
            } else {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                return;
            }
        }

        void op_num_assign::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X(1) == x) {
                Set input;
                input.add(dy);
                op_num_assign num_assign;
                num_assign.f(input, output, data);
                data.at(output(1)).setName("dx = dy");
            } else {
                op_num_constant numConstant;
                numConstant.f(X, output, data);
                return;
            }
        }

        void op_num_assign::f(const Set &X, Set &output, VariableMap &data) {
            MATH21_ASSERT(X.size() == 1)
            NumN y = data.createV("op_num_assign(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_num_assign::fv(const Set &X, const Set &Y, VariableMap &data) const {
            if (X.isEmpty()) {
                return;
            }
            NumN y = Y(1);
            if(data.at(y).isComputed()){
                return;
            }
            data.at(y).getValue().setSize(1);
            data.at(y).getValue() = data(X(1)).getValue()(1);;
            data.at(y).setComputed(1);
        }

        void op_num_assign::setSize(const Set &X, const Set &Y, VariableMap &data) const {
        }

        Function *op_num_assign::clone() const {
            Function *f = new op_num_assign();
            return f;
        }

        const char *op_num_assign::getName() const {
            return "op_num_assign";
        }
    }
}