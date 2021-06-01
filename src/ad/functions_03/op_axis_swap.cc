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
#include "op_axis_swap.h"

namespace math21 {
    namespace ad {
        NumN op_axis_swap::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos == 1)
            op_axis_swap _;
            Function &f = _;
            Set X2;
            X2.add(dy);
            X2.add(X(2));
            X2.add(X(3));
            NumN dx;
            dx = f.evaluate(X2, data);
            return dx;
        }

        // X = {x, pos, pos2}
        void op_axis_swap::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(X.size() == 3);
            NumN y = Y(1);
            auto &y_value = data.at(y).getValue();
            auto &x = data(X(1)).getValue();
            auto &pos_value = data(X(2)).getValue();
            MATH21_ASSERT(pos_value.isScalarInMath());
            NumZ pos = (NumZ) pos_value(1);
            auto &pos2_value = data(X(3)).getValue();
            MATH21_ASSERT(pos2_value.isScalarInMath());
            NumZ pos2 = (NumZ) pos2_value(1);
            math21_op_tensor_swap_axes(x, y_value, pos, pos2);
        }
    }
}