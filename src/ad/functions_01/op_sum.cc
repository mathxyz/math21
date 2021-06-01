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
#include "../functions_02/op_broadcast_tensor.h"
#include "op_multiply.h"
#include "op_share_reshape.h"
#include "op_sum.h"

namespace math21 {
    namespace ad {
        NumN op_sum::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1 || X.size() == 2);
            math21_tool_assert(X(1) == x);
            const auto &x_value = data.at(x).getValue();
            const auto &dy_value = data.at(dy).getValue();
            NumN dx;
            if (X.size() == 1) {
                math21_tool_assert(dy_value.size() == 1);
                NumN k = data.createC("1");
                variable_set_device_type_using_variable(X(1), k, data);
                auto &k_vec = data.at(k).getValue();
                k_vec.setSize(x_value.size());
                k_vec = 1;
                // todo: use op_scalar_mul to speed up
                op_multiply multiply0;
                Function &multiply = multiply0;
                dx = multiply.evaluate(dy, k, data);
            } else {
                VecN d;
                x_value.shape(d);
                NumN k = data.createC("shape");
                data.at(k).getValue() = d;
                op_broadcast_tensor _op_broadcast_tensor;
                Function &f_op_broadcast_tensor = _op_broadcast_tensor;
                VecN axes;
                axes = data.at(X(2)).getValue();
                VecN d_y;
                // todo: maybe put to graph
                math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(
                        data(x).getValue(), axes, d_y);

                NumN k_d = data.createC("shape");
                data.setValue(k_d, 1);
                data.at(k_d).getValue() = d_y;
                op_share_reshape _op_share_reshape;
                Function &f_op_share_reshape = _op_share_reshape;
                dy = f_op_share_reshape.evaluate(dy, k_d, data);
                dx = f_op_broadcast_tensor.evaluate(dy, k, data);
            }
            return dx;
        }

        // X = {x, axes}
        void op_sum::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1 || X.size() == 2);
            const auto &x_value = data.at(X(1)).getValue();
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            VecN axes;
            if (X.size() == 1) {
//                y_value.setSize(1);
//                y_value = math21_operator_container_sum(x_value, 1);
            } else {
                axes = data.at(X(2)).getValue();
//                math21_operator_tensor_sum_along_axes(x_value, y_value, axes, isKeepingDims);
            }
            math21_op_sum(x_value, y_value, axes, isKeepingDims);
        }
    }
}