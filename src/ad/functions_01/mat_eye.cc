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

namespace math21 {
    namespace ad {
        op_mat_eye::op_mat_eye() {
        }

        op_mat_eye::~op_mat_eye() {
        }

        void op_mat_eye::df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const {
            output.clear();
        }

        void op_mat_eye::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            output.clear();
            return;
        }

        void op_mat_eye::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            output.clear();
            return;
        }

        void op_mat_eye::f(const Set &X, Set &output, VariableMap &data) {
            MATH21_ASSERT(X.size() == 1)
            NumN y = data.createV("op_mat_eye(x)");
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_mat_eye::fv(const Set &X, const Set &Y, VariableMap &data) const {
            if (X.isEmpty()) {
                return;
            }
            NumN x1 = X(1);
            NumN y = Y(1);
            if(data.at(y).isComputed()){
                return;
            }
            const auto &x1_vec = data(x1).getValue();
            auto &y_mat = data.at(y).getValue();
            y_mat.setSize(x1_vec.size(), x1_vec.size());
            math21_operator_mat_eye(y_mat);
            data.at(Y(1)).setComputed(1);
        }

        void op_mat_eye::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            fv(X, Y, data);
            if (!Variable::isRequestingAbstractCompletely()) {
                data.at(Y(1)).synchronizeToZero(data);
            }
            data.at(Y(1)).setComputed(1);
        }

        void op_mat_eye::setSize(const Set &X, const Set &Y, VariableMap &data) const {
        }

        Function *op_mat_eye::clone() const {
            Function *f = new op_mat_eye();
            return f;
        }

        const char *op_mat_eye::getName() const {
            return "op_mat_eye";
        }
    }
}