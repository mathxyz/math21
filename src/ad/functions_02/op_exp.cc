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
#include "op_exp.h"

namespace math21 {
    namespace ad {
        NumN op_exp::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            op_exp exp0;
            Function &exp = exp0;
            return exp.evaluate(x, data);
        }

        void op_exp::df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            op_exp num_exp;
            Function &function = num_exp;
            function.f(x, dydx, data);
        }

        void op_exp::df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            op_exp num_exp;
            Function &function = num_exp;
            function.forward(x, dydx, data);
        }

//        NumR op_exp::evaluate_at_num(NumR x) const {
//            return xjexp(x);
//        }

        void op_exp::evaluate_at_tensor(const VecR &x, VecR &y) const {
            math21_op_exp(x, y);
        }

    }
}