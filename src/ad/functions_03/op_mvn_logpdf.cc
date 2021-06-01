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
#include "op_mvn_logpdf.h"

namespace math21 {
    namespace ad {
        // Note: dY/dX is just diag(dY/dX)
        // y = f(x) => dy/dx = exp(x-y)
        NumN op_mvn_logpdf::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(X.size() == 3);
            NumN pos = math21_operator_container_arg(X, x);
            op_mvn_dlogpdf dlogpdf(pos);
            Function &f = dlogpdf;
            return f.evaluate(X, data);
        }

        NumN op_mvn_logpdf::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            NumN dydx = df_vjp(X, x, y, data);
            NumN dx;

            if (pos == 1) {
                NumN dy_v = dy;
                {
                    const auto &dy_value = data(dy).getValue();
                    if (dy_value.size() > 1 && dy_value.isRowVector()) {
                        op_mat_trans matTrans;
                        Function &f_trans = matTrans;
                        dy_v = f_trans.evaluate(dy, data);
                    }
                }
                op_multiply multiply0;
                Function &multiply = multiply0;
                dx = multiply.evaluate(dy_v, dydx, data);
            } else {
                NumN dy_h = dy;
                {
                    const auto &dy_value = data(dy).getValue();
                    if (dy_value.size() > 1 && dy_value.isColVector()) {
                        op_mat_trans matTrans;
                        Function &f_trans = matTrans;
                        dy_h = f_trans.evaluate(dy, data);
                    }
                }

                op_mat_mul multiply0;
                Function &multiply = multiply0;
                dx = multiply.evaluate(dy_h, dydx, data);
            }
            return dx;
        }

        // support x and batch X
        // x is vector if x is not batched.
        // x is matrix if x is batched.
        void op_mvn_logpdf::evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const {
            const VecR &x = x1;
            const VecR &mean = x2;
            const MatR &covariance = x3;
            if (!math21_pr_mvn_logpdf2(x, mean, covariance, y)) {
                math21_tool_assert(0);
            }
        }

        NumN op_mvn_dlogpdf::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
math21_tool_assert(0 && "not support!");
            return 0;
        }

        void op_mvn_dlogpdf::evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const {
            const VecR &x = x1;
            const VecR &mean = x2;
            const MatR &covariance = x3;
            if (pos == 1) {
                math21_pr_mvn_dYdX_diag_logpdf(x, mean, covariance, y);
            } else if (pos == 2) {
                math21_pr_mvn_dYdmu_logpdf(x, mean, covariance, y);
            } else if (pos == 3) {
                math21_pr_mvn_dYdSig_logpdf(x, mean, covariance, y);
                if (y.dims() > 2) {
                    NumN nr = y.dim(1);
                    NumN nc = y.size() / nr;
                    VecN d(2);
                    d = nr, nc;
                    // can reshape here, because y is not exposed outside.
                    math21_operator_tensor_reshape_to_same_vspace_using_shape(d, y);
                } else {
                    m21warn("Check here please!");
                }
            } else {
                math21_tool_assert(0);
            }
        }
    }
}