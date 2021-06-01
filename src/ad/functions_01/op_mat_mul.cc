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
#include "num_add.h"
#include "num_multiply.h"
#include "mat_jacobian.h"
#include "op_mat_mul.h"
#include "op_mat_trans.h"

namespace math21 {
    namespace ad {
        /*
         z = f(Y), Y = W*X
         f'(Y) => dz/dW, dz/dX

         shape requirement: z: l, W: m*n, X: n*b, Y: m*b
         f'(Y): l*(m*b)
         dz/dW: l*(m*n)
         dz/dX: l*(n*b)

         z, Y, W, X can be tensors, but must be reshaped to the shape required beforehand.
         dY := dz/dY = f'(Y), shape: l*(m*b)
         didY := f'(Y)(i) := dz(i)/dY
         f'(Y)(i) shape: 1*(m*b)
         f'(Y)(i).as_mat shape: m*b
         dY/dW = diag(X.trans), shape: (m*b)*(m*n), or m*m with X_trans having shape b*n
         dY/dX = diag(W).(trans*trans), shape: (m*b)*(n*b)
         =>
         dz(i)/dW = dz(i)/dY * dY/dW
         dz(i)/dW = f'(Y)(i).as_mat*X.trans, shape as mat: m*n, for any i in {1, ..., l}
         dz/dW = (dz(1)/dW, ..., dz(i)/dW, ..., dz(l)/dW), shape: l*(m*n)

         dz(i)/dX = dz(i)/dY * dY/dX
         dz(i)/dX = W.trans*f'(Y)(i), shape as mat: n*b
         dz/dX = (dz(1)/dX, ..., dz(i)/dX, ..., dz(l)/dX), shape: l*(n*b)
         * */

        // data is for debug only.
        // Y = W*X
        template<typename T>
        void math21_ad_mat_mul_fv(const Tensor<T> &W, const Tensor<T> &X, Tensor<T> &Y, NumB isTransW = 0,
                                  NumB isTransX = 0, VariableMap *data = 0) {
            if (data) {
                MATH21_ASSERT(W.dims() == 2 && X.dims() == 2 && W.dim(2) == X.dim(1),
                              "" << W.log("W") << X.log("X") << Y.log("Y"));
//                <<data->log("data")
            } else {
                MATH21_ASSERT(W.dims() == 2 && X.dims() == 2 && W.dim(2) == X.dim(1),
                              "" << W.log("W") << X.log("X") << Y.log("Y"))
            }
//            MATH21_ASSERT(Y.dim(1) == W.dim(1) && Y.dim(2) == X.dim(2))
//            math21_operator_matrix_mul_with_trans_option(1, W, X, Y, isTransW, isTransX);
            math21_op_mat_mul(1, W, X, Y, isTransW, isTransX);
        }

        // dz(i)/dW = f'(Y)(i).as_mat*X.trans for any i
        // dz/dW
        // f'(Y): l*(m*b)
        // dz/dW: l*(m*n)
        // X: n*b
        // dz/dX: l*(n*b)
        template<typename T>
        void math21_ad_mat_mul_dW_fv(const Tensor<T> &W, const Tensor<T> &X, const Tensor<T> &dY, Tensor<T> &dW) {
            MATH21_ASSERT(W.dims() == 2 && X.dims() == 2 && W.dim(2) == X.dim(1),
                          "" << W.log("W") << X.log("X") << dY.log("dY"))
            Seqce<Tensor<T>> dYs;
            Seqce<Tensor<T>> dWs;
//                W.log("W");
//                X.log("X");
//                dY.log("dY");
//                math21_operator_share_reshape_remove_dim_1(dY, dYs);
            math21_operator_share_reshape_mat_2_mats(dY, dYs, W.dim(1), X.dim(2));
//                dYs.log("dYs");
            dW.setSize(dY.dim(1), W.size());
            math21_operator_share_reshape_mat_2_mats(dW, dWs, W.dim(1), W.dim(2));

            for (NumN i = 1; i <= dY.dim(1); ++i) {
                if (!dY.isScalarInMath()) {
                    math21_ad_mat_mul_fv(dYs(i), X, dWs(i), 0, 1);
                }
            }
        }

        template<typename T>
        void math21_ad_mat_mul_dX_fv(const Tensor<T> &W, const Tensor<T> &X, const Tensor<T> &dY, Tensor<T> &dX) {
            MATH21_ASSERT(W.dims() == 2 && X.dims() == 2 && W.dim(2) == X.dim(1),
                          "" << W.log("W") << X.log("X") << dY.log("dY"))
            Seqce<Tensor<T>> dYs;
            Seqce<Tensor<T>> dXs;
//                W.log("W");
//                X.log("X");
//                dY.log("dY");
//                math21_operator_share_reshape_remove_dim_1(dY, dYs);
            math21_operator_share_reshape_mat_2_mats(dY, dYs, W.dim(1), X.dim(2));
//                dYs.log("dYs");
            dX.setSize(dY.dim(1), X.size());
            math21_operator_share_reshape_mat_2_mats(dX, dXs, X.dim(1), X.dim(2));

            for (NumN i = 1; i <= dY.dim(1); ++i) {
                math21_ad_mat_mul_fv(W, dYs(i), dXs(i), 1, 0);
            }
        }

        op_mat_mul::op_mat_mul() {
        }

        op_mat_mul::~op_mat_mul() {
        }

        NumN op_mat_mul::cr_vjp_inner(const Set &X, NumN x0, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            math21_tool_assert(X.contains(x0));

            op_mat_trans mat_trans0;
            Function &mat_trans = mat_trans0;
            op_mat_mul mat_mul0;
            Function &mat_mul = mat_mul0;

            // data in dy can be matrix, see math21_operator_matrix_ad_reverse_mul
            NumN w = X(1);
            NumN x = X(2);
            if (x0 == w) {
                NumN x_trans = mat_trans.evaluate(x, data);
                return mat_mul.evaluate(dy, x_trans, data);
            } else {
                NumN w_trans = mat_trans.evaluate(w, data);
                return mat_mul.evaluate(w_trans, dy, data);
            }
        }

        NumN op_mat_mul::evaluate(const Set &X, VariableMap &data) {
            math21_tool_assert(X.size() == 2);
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

        void op_mat_mul::fv(const Set &X, const Set &Y, VariableMap &data) const {
            NumN y = Y(1);
            NumN w = X(1);
            NumN x = X(2);
            math21_ad_mat_mul_fv(data(w).getValue(),
                                 data(x).getValue(),
                                 data.at(y).getValue(),
                                 0, 0, &data);
        }

        void op_mat_mul::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X.size() == 0) {
                return;
            }
            MATH21_ASSERT(X.size() == 2, "not implemented")
            if (Variable::isRequestingAbstractCompletely()) {
                if (x == X(1)) {
                    op_mat_dmul dmul(1);
                    Set input;
                    input.add(X);
                    input.add(dy);
                    dmul.f(input, output, data);
                } else {
                    op_mat_dmul dmul(2);
                    Set input;
                    input.add(X);
                    input.add(dy);
                    dmul.f(input, output, data);
                }
            } else {
                NumN x_jacobian;
                std::string name;
                if (x == X(1)) {
                    x_jacobian = X(1);
                    name = "dx = dy * d1(op_mat_mul(x))";
                } else {
                    x_jacobian = X(2);
                    name = "dx = dy * d2(op_mat_mul(x))";
                }
                MATH21_ASSERT(data(y).isComputed())
                op_mat_jacobian jacobian;
                Set X_jacobian;
                X_jacobian.add(x_jacobian);
                X_jacobian.add(y);
                jacobian.f(X_jacobian, output, data);

                Set input;
                input.add(dy);
                input.add(output);
                op_mat_mul mat_mul;
                mat_mul.f(input, output, data);
                data.at(output(1)).setName(name.c_str());
            }
        }

        void op_mat_mul::backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            if (X.size() == 0) {
                return;
            }
            MATH21_ASSERT(X.size() == 2, "not implemented")
            if (Variable::isRequestingAbstractCompletely()) {
                if (x == X(1)) {
                    op_mat_dmul dmul0(1);
                    Function &dmul = dmul0;
                    Set input;
                    input.add(X);
                    input.add(dy);
                    dmul.forward(input, output, data);
                } else {
                    op_mat_dmul dmul0(2);
                    Function &dmul = dmul0;
                    Set input;
                    input.add(X);
                    input.add(dy);
                    dmul.forward(input, output, data);
                }
            } else {
                NumN x_jacobian;
                std::string name;
                if (x == X(1)) {
                    x_jacobian = X(1);
                    name = "dx = dy * d1(op_mat_mul(x))";
                } else {
                    x_jacobian = X(2);
                    name = "dx = dy * d2(op_mat_mul(x))";
                }
                MATH21_ASSERT(data(y).isComputed())
                op_mat_jacobian jacobian0;
                Function &jacobian = jacobian0;
                Set X_jacobian;
                X_jacobian.add(x_jacobian);
                X_jacobian.add(y);
                jacobian.forward(X_jacobian, output, data);

                Set input;
                input.add(dy);
                input.add(output);
                op_mat_mul mat_mul0;
                Function &mat_mul = mat_mul0;
                mat_mul.forward(input, output, data);
                data.at(output(1)).setName(name.c_str());
            }
        }

        void op_mat_mul::f(const Set &X, Set &output, VariableMap &data) {
            if (X.size() == 0) {
                return;
            }
            if (X.size() == 1) {
                output.clear();
                output.add(X(1));
                return;
            }
            MATH21_ASSERT(X.size() == 2, "not implemented")
            NumN y = data.createV(math21_string_concatenate(getName(), "(x)").c_str());
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        void op_mat_mul::compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) {
            if (Variable::isRequestingAbstractCompletely()) {
                fv(X, Y, data);
                return;
            }
            MATH21_ASSERT (X.size() == 2);

            NumN w = X(1);
            NumN x = X(2);
            NumN y = Y(1);
            const auto &w_mat = data(w).getVariableMat();
            const auto &x_mat = data(x).getVariableMat();
            MATH21_ASSERT (!data(w).isAbstractCompletely());
            MATH21_ASSERT (!data(x).isAbstractCompletely());

            MATH21_ASSERT(w_mat.dims() == 2 && x_mat.dims() == 2 && w_mat.dim(2) == x_mat.dim(1),
                          "" << w_mat.log("W") << x_mat.log("X"))

            if (data.at(y).isAbstractCompletely()) {
                data.at(y).setAbstractZero();
                auto &y_mat = data.at(y).getVariableMat();
                y_mat.setSize(w_mat.dim(1), x_mat.dim(2));
                for (NumN i = 1; i <= y_mat.dim(1); ++i) {
                    for (NumN j = 1; j <= y_mat.dim(2); ++j) {
                        op_num_add num_add;
                        Set X_add;
                        for (NumN k = 1; k <= x_mat.dim(1); ++k) {
                            op_num_multiply multiply0;
                            Function &multiply = multiply0;
                            NumN y_multiply;
                            multiply.f(w_mat(i, k), x_mat(k, j), y_multiply, data);
                            X_add.add(y_multiply);
                            std::string name = math21_string_concatenate(getName(),
                                                                         math21_string_to_string(i),
                                                                         math21_string_to_string(j),
                                                                         math21_string_to_string(k),
                                                                         "(x)");
                            data.at(y_multiply).setName(name.c_str());
                        }
                        Set Y_add;
                        num_add.f(X_add, Y_add, data);
                        y_mat(i, j) = Y_add(1);
                        std::string name = math21_string_concatenate(getName(),
                                                                     math21_string_to_string(i),
                                                                     math21_string_to_string(j),
                                                                     "(x)");
                        data.at(y_mat(i, j)).setName(name.c_str());
                    }
                }
            } else {
                auto &y_mat = data.at(y).getVariableMat();
                MATH21_ASSERT(y_mat.isSameSize(w_mat.dim(1), x_mat.dim(2)));
            }

            auto &y_mat = data.at(y).getVariableMat();
            for (NumN i = 1; i <= y_mat.dim(1); ++i) {
                for (NumN j = 1; j <= y_mat.dim(2); ++j) {
                    derivative.compute(y_mat(i, j));
                }
            }
            data.at(y).synchronizeValue(data);
            data.at(Y(1)).setComputed(1);
        }

        void op_mat_mul::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented");
        }

        Function *op_mat_mul::clone() const {
            Function *f = new op_mat_mul();
            return f;
        }

        const char *op_mat_mul::getName() const {
            return "op_mat_mul";
        }
    }

    namespace ad {
        op_mat_dmul::op_mat_dmul(NumN varId) : _varId(varId) {
        }

        op_mat_dmul::~op_mat_dmul() {
        }

        void op_mat_dmul::cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const {
            MATH21_ASSERT(0)
        }

        void op_mat_dmul::f(const Set &X, Set &output, VariableMap &data) {
            MATH21_ASSERT(X.size() == 3, "not implemented")
            std::string name;
            if (_varId == 1) {
                name = math21_string_concatenate(getName(), "1(x)");
            } else {
                name = math21_string_concatenate(getName(), "2(x)");
            }
            NumN y = data.createV(name.c_str());
            data.at(y).setf(this);
            data.at(y).setX(X);
            for (NumN i = 1; i <= X.size(); ++i) {
                data.at(X(i)).addy(y);
            }
            output.clear();
            output.add(y);
        }

        // dot product
        void op_mat_dmul::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(X.size() == 3)
            if (_varId == 1) {
                math21_ad_mat_mul_dW_fv(data(X(1)).getValue(),
                                        data(X(2)).getValue(),
                                        data(X(3)).getValue(),
                                        data.at(Y(1)).getValue());
            } else {
                math21_ad_mat_mul_dX_fv(data(X(1)).getValue(),
                                        data(X(2)).getValue(),
                                        data(X(3)).getValue(),
                                        data.at(Y(1)).getValue());
            }

            // maybe can be used for when Y(1).dim = 1
//            math21_operator_matrix_ad_reverse_mul(data(X(1)).getValue(),
//                                                  data(X(2)).getValue(),
//                                                  data(X(3)).getValue(),
//                                                  data.at(Y(1)).getValue(), _varId);

            data.at(Y(1)).setComputed(1);
        }

        void op_mat_dmul::setSize(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(0, "not implemented");
        }

        Function *op_mat_dmul::clone() const {
            Function *f = new op_mat_dmul(_varId);
            return f;
        }

        const char *op_mat_dmul::getName() const {
            return "op_mat_dmul";
        }
    }
}