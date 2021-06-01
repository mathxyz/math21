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

#include "files.h"

namespace math21 {
    namespace ad {
        op_num_constant::op_num_constant() {
        }

        op_num_constant::~op_num_constant() {
        }

        void op_num_constant::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            Set input;
            op_num_constant numConstant;
            numConstant.f(input, output, data);
        }

        void op_num_constant::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            Set input;
            op_num_constant numConstant;
            numConstant.f(input, output, data);
        }

        void op_num_constant::f(NumN &y, VariableMap &data) {
            Set X;
            Set Y;
            f(X, Y, data);
            y = Y(1);
        }

        // todo: if error, maybe use createV
        void op_num_constant::f(const Set &X, Set &output, VariableMap &data) {
            NumN y = data.createC("op_num_constant(x)");
            data.setValue(y, 0);
            data.at(y).setf(this);
            output.clear();
            output.add(y);
        }

        void op_num_constant::fv(const Set &X, const Set &Y, VariableMap &data) const {
        }

        void op_num_constant::setSize(const Set &X, const Set &Y, VariableMap &data) const {
        }

        Function *op_num_constant::clone() const {
            Function *f = new op_num_constant();
            return f;
        }

        const char *op_num_constant::getName() const {
            return "op_num_constant";
        }
    }
}