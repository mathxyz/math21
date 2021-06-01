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
#include "op_power.h"

namespace math21 {
    namespace ad {
        op_power::op_power(NumR k, NumR p) : k(k), p(p) {
        }

        op_power::~op_power() {
        }

        NumN op_power::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1);
            math21_tool_assert(X(1) == x);
            if (k == 0 || p == 0) {
                auto &dy_vec = data.at(dy).getValue();
                NumN dx = data.createC("0");
                variable_set_device_type_using_variable(X(1), dx, data);
                auto &dx_vec = data.at(dx).getValue();
                dx_vec.setSize(dy_vec.size());
                dx_vec = 0; // todo: consider using ad_global_get_constant_0() + broadcast.
                return dx;
            }
            op_power power0(k * p, p - 1);
            Function &power = power0;
            NumN y_power = power.evaluate(x, data);
            op_multiply multiply0;
            Function &multiply = multiply0;
            return multiply.evaluate(dy, y_power, data);
        }

        void op_power::df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            if (k == 0 || p == 0) {
                dydx = ad_global_get_constant_0();
                return;
            }
            NumN y_power;
            op_power power0(k * p, p - 1);
            Function &power = power0;
            power.f(x, y_power, data);
            dydx = y_power;
        }

        void op_power::df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const {
            if (X(1) != x) {
                dydx = ad_global_get_constant_0();
                return;
            }

            if (k == 0 || p == 0) {
                dydx = ad_global_get_constant_0();
                return;
            }
            NumN y_power;
            op_power power0(k * p, p - 1);
            Function &power = power0;
            power.forward(x, y_power, data);
            dydx = y_power;
        }

//        NumR op_power::evaluate_at_num(NumR x) const {
//            return xjmultiply(k, xjpow(x, p));
//        }

        void op_power::evaluate_at_tensor(const VecR &x, VecR &y) const {
            math21_op_pow(x, p, y);
            math21_op_mul_onto(k, y);
        }

        Function *op_power::clone() const {
            Function *f = new op_power(k, p);
            return f;
        }

        const char *op_power::getName() const {
            return "op_power";
        }
    }
}