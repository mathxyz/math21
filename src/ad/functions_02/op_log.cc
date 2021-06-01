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

#include "op_power.h"
#include "inner_cc.h"
#include "op_log.h"

namespace math21 {
    namespace ad {
        op_log::op_log() {
            isBaseEulersNumber = 1;
            a = 0;
        }

        op_log::op_log(NumR a) : a(a) {
            isBaseEulersNumber = 0;
        }

        op_log::~op_log() {
        }

        NumN op_log::df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            NumR k;
            if (isBaseEulersNumber) {
                k = 1;
            } else {
                k = xjdivide((NumR) 1.0, xjlog(a));
            }
            op_power power0(k, -1);
            Function &power = power0;
            return power.evaluate(x, data);
        }

        // (a^x)' = ln(a) * a^x
        // log(a, x)' = (1/ln(a)) * (1/x)
        void op_log::df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            NumR k;
            if (isBaseEulersNumber) {
                k = 1;
            } else {
                k = xjdivide((NumR) 1.0, xjlog(a));
            }
            op_power num_power(k, -1);
            Function &function = num_power;
            function.f(x, dydx, data);
        }

        void op_log::df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            NumR k;
            if (isBaseEulersNumber) {
                k = 1;
            } else {
                k = xjdivide((NumR) 1.0, xjlog(a));
            }
            op_power num_power(k, -1);
            Function &function = num_power;
            function.forward(x, dydx, data);
        }

//        NumR op_log::evaluate_at_num(NumR x) const {
//            if (isBaseEulersNumber) {
//                return xjlog(x);
//            } else {
//                return xjdivide(xjlog(x), xjlog(a));
//            }
//        }

        void op_log::evaluate_at_tensor(const VecR &x, VecR &y) const {
            math21_op_log(x, y);
            if (!isBaseEulersNumber) {
                math21_op_divide_onto(y, xjlog(a));
            }
        }

        Function *op_log::clone() const {
            op_log *f = new op_log();
            f->a = a;
            f->isBaseEulersNumber = isBaseEulersNumber;
            return f;
        }

        const char *op_log::getName() const {
            if (isBaseEulersNumber) {
                return "op_num_ln";
            } else if (a == 2) {
                return "op_log2";
            } else if (a == 10) {
                return "op_log10";
            } else {
                return "op_log";
            }
        }
    }
}