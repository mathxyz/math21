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
#include "op_broadcast_tensor.h"

namespace math21 {
    namespace ad {

        NumN op_broadcast_tensor::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            math21_tool_assert(X(1) == x);

            const auto &x_value = data.at(X(1)).getValue();
            const auto &d_value = data.at(X(2)).getValue();
            VecR axes;
            VecN d;
            d = d_value;
            VecN d_x(d.size());
            d_x = 1;
            math21_operator_container_set_partially(x_value.shape(), d_x, 0, 0, x_value.dims());
            math21_operator_container_arg_not_equal(d_x, d, axes, 0);
            if (axes.isEmpty()) {
                op_assign assign0;
                Function &assign = assign0;
                return assign.evaluate(dy, data);
            } else {
                NumN k = data.createC("axes");
                auto &k_vec = data.at(k).getValue();
                k_vec = axes;
                op_sum sum0;
                Function &sum = sum0;
                return sum.evaluate(dy, k, data);
            }
        }

        void op_broadcast_tensor::fv(const Set &X, const Set &Y, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            NumN y = Y(1);
            const auto &x_value = data.at(X(1)).getValue();
            const auto &d_value = data.at(X(2)).getValue();
            MATH21_ASSERT(d_value.dims() == 1)
            VecN d;
            d = d_value;
            auto &y_value = data.at(y).getValue();
            math21_op_tensor_broadcast(x_value, y_value, d); // todo: make y sparse.
        }

        NumN op_create::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 2);
            math21_tool_assert(X(1) == x);
            op_sum _;
            Function &f = _;
            return f.evaluate(dy, data);
        }

        // X = {x, d}.
        // x is a number; d is shape.
        void op_create::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(X.size() == 2);
            const auto &x = data.at(X(1)).getValue();
            MATH21_ASSERT(x.isScalarInMath());
            VecR x_cpu;
            x_cpu = x;
            const auto &d_value = data.at(X(2)).getValue();
            MATH21_ASSERT(d_value.dims() == 1)
            VecN d;
            d = d_value;
            auto &y = data.at(Y(1)).getValue();
            y.setSize(d);
            math21_op_vector_set_value(y, x_cpu(1));
        }
    }
}