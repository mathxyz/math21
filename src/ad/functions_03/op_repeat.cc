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
#include "op_repeat.h"

namespace math21 {
    namespace ad {
        NumN op_repeat::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos == 1)
            op_undo_repeat_sum _;
            Function &f = _;
            Set X2;
            X2.add(dy);
            X2.add(X(2));
            X2.add(X(3));
            NumN dx;
            dx = f.evaluate(X2, data);
            return dx;
        }

        void op_repeat::evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const {
            const VecR &x = x1;
            VecN repeats;
            repeats = x2;
            NumZ axis;
            MATH21_ASSERT(x3.isScalarInMath())
            axis = (NumZ) x3(1);
            math21_op_tensor_repeat(x, y, repeats, axis);
        }

        NumN op_undo_repeat_sum::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            NumN pos = math21_operator_container_arg(X, x);
            MATH21_ASSERT(pos == 1)
            op_repeat _;
            Function &f = _;
            Set X2;
            X2.add(dy);
            X2.add(X(2));
            X2.add(X(3));
            NumN dx;
            dx = f.evaluate(X2, data);
            return dx;
        }

        void op_undo_repeat_sum::evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const {
            const VecR &x = x1;
            VecN repeats;
            repeats = x2;
            NumZ axis;
            MATH21_ASSERT(x3.isScalarInMath())
            axis = (NumZ) x3(1);
            math21_op_tensor_sum_undo_repeat(y, x, repeats, axis);
        }
    }
}