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
#include "op_logsumexp.h"

namespace math21 {
    namespace ad {
        op_logsumexp::op_logsumexp(const VecN &axes, NumB isKeepingDims)
                : isKeepingDims(isKeepingDims) {
            this->axes = axes;
        }

        // y = f(x) => dy/dx = exp(x-y)
        NumN op_logsumexp::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            NumN k = data.createC("-1");
            variable_set_device_type_using_variable(x, k, data);
            data.setValue(k, -1);

            op_multiply mul0;
            Function &mul = mul0;
            NumN my = mul.evaluate(k, y, data);

            VecN d_y;
            // todo: maybe put to graph
            math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(
                    data(x).getValue(), axes, d_y);

            NumN k_d = data.createC("shape");
            data.at(k_d).getValue() = d_y;
            op_share_reshape _op_share_reshape;
            Function &f_op_share_reshape = _op_share_reshape;
            my = f_op_share_reshape.evaluate(my, k_d, data);

            op_add add0;
            Function &add = add0;
            NumN xmy = add.evaluate(x, my, data);
            op_exp exp0;
            Function &exp = exp0;
            NumN df = exp.evaluate(xmy, data);
            auto &value_df = data.at(df).getValue();
            MATH21_ASSERT_CODE(value_df.isSameSize(data(x).getValue().shape()))
            return df;
        }

        NumN op_logsumexp::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN dydx = df_vjp(X, x, y, data);
            NumN dx;

            VecN d_y;
            // todo: maybe put to graph
            math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(
                    data(x).getValue(), axes, d_y);

            NumN k_d = data.createC("shape");
            data.at(k_d).getValue() = d_y;
            op_share_reshape _op_share_reshape;
            Function &f_op_share_reshape = _op_share_reshape;
            dy = f_op_share_reshape.evaluate(dy, k_d, data);

            op_multiply multiply0;
            Function &multiply = multiply0;
            dx = multiply.evaluate(dy, dydx, data);
            return dx;
        }

        void op_logsumexp::evaluate_at_vec(const VecR &x, VecR &y) const {
            math21_op_logsumexp(x, y, axes, isKeepingDims);
//            math21_operator_tensor_f_along_axes(x, y, math21_operator_vector_logsumexp, axes, isKeepingDims);
        }
    }
}