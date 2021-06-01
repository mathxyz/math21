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
#include "op_mat_mul.h"
#include "mat_jacobian.h"

namespace math21 {
    namespace ad {
        op_mat_jacobian::op_mat_jacobian() {
        }

        op_mat_jacobian::~op_mat_jacobian() {
        }

        void op_mat_jacobian::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            // y must be computed in define-by-run mode.
            MATH21_ASSERT(data(y).isComputed())
            MATH21_ASSERT(X.size() == 2)
            if (X(2) == x) {
                output.clear();
                return;
            }
            op_mat_jacobian mat_jacobian;
            Set X_jacobian;
            X_jacobian.add(X(1));
            X_jacobian.add(y);
            mat_jacobian.f(X_jacobian, output, data);

            Set input;
            input.add(dy);
            input.add(output);
//                MATH21_ASSERT(0)
            op_mat_mul mul;
            mul.f(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_mat_jacobian(x))");
        }

        void op_mat_jacobian::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            // y must be computed in define-by-run mode.
            MATH21_ASSERT(data(y).isComputed())
            MATH21_ASSERT(X.size() == 2)
            if (X(2) == x) {
                output.clear();
                return;
            }
            op_mat_jacobian mat_jacobian;
            Set X_jacobian;
            X_jacobian.add(X(1));
            X_jacobian.add(y);
            mat_jacobian.forward(X_jacobian, output, data);

            Set input;
            input.add(dy);
            input.add(output);
//                MATH21_ASSERT(0)
            op_mat_mul mul;
            mul.forward(input, output, data);
            data.at(output(1)).setName("dx = dy * d(op_mat_jacobian(x))");
        }

        void op_mat_jacobian::f(const Set &X, Set &output, VariableMap &data) {
            NumN jacobian = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(jacobian).setf(this);
            data.at(jacobian).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(jacobian);
            }
            output.clear();
            output.add(jacobian);

            MATH21_ASSERT (X.size() == 2);

            NumN x = X(1);
            NumN y = X(2);
            const auto &x_vec = data(x).getVariableMat();
            const auto &y_vec = data(y).getVariableMat();
            MATH21_ASSERT (!data(x).isAbstractCompletely());
            MATH21_ASSERT (!data(y).isAbstractCompletely());

            data.at(jacobian).setAbstractZero();
            auto &jacobian_mat = data.at(jacobian).getVariableMat();
            jacobian_mat.setSize(y_vec.size(), x_vec.size());

            Derivative derivative(data);
//            math21::timer t;
//            t.start();
            for (NumN i = 1; i <= y_vec.size(); ++i) {
                Map DT;
                for (NumN j = 1; j <= x_vec.size(); ++j) {
                    NumN dij = derivative.cd(x_vec(j), y_vec(i), DT);
                    if (dij != 0) {
                        std::string name = math21_string_concatenate(getName(),
                                                                     math21_string_to_string(i),
                                                                     math21_string_to_string(j),
                                                                     "(x)");
                        data.at(dij).setName(name.c_str());
                    } else {
                        dij = ad_global_get_constant_0();
                    }
                    jacobian_mat(i, j) = dij;
                }
            }
//            t.end();
//            math21::m21log(math21_string_concatenate(getName(), " time "), t.time());
        }

        // dot product
        void op_mat_jacobian::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void op_mat_jacobian::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            if(data(Y(1)).isComputed()){
                return;
            }
            auto &jacobian_mat = data.at(Y(1)).getVariableMat();
            for (NumN i = 1; i <= jacobian_mat.nrows(); ++i) {
                for (NumN j = 1; j <= jacobian_mat.ncols(); ++j) {
//                    Variable &vij = data.at(jacobian_mat(i, j));
//                    Function &f = vij.getf();
//                    Set output_ele;
//                    output_ele.add(jacobian_mat(i, j));
//                    f.compute(vij.getX(), output_ele, data, derivative);
                    derivative.compute(jacobian_mat(i, j));
                }
            }
            data.at(Y(1)).synchronizeValue(data);
            data.at(Y(1)).setComputed(1);
        }

        void op_mat_jacobian::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented");
        }

        Function *op_mat_jacobian::clone() const {
            Function *f = new op_mat_jacobian();
            return f;
        }

        const char *op_mat_jacobian::getName() const {
            return "op_mat_jacobian";
        }
    }
}