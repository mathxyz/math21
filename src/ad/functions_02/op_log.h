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

#pragma once

#include "inner.h"

namespace math21 {
    namespace ad {
        // logarithm function log(a, x)
        struct op_log : public op_ele_unary {
        private:
            NumR a; // a is base
            NumB isBaseEulersNumber; // natural logarithm, ln(x)
        public:
            op_log();

            op_log(NumR a);

            virtual ~op_log();

            NumN df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const override;

            void df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const override;

            void df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const override;

//            NumR evaluate_at_num(NumR x) const override;
            void evaluate_at_tensor(const VecR &x, VecR &y) const override;

            Function *clone() const override;

            const char *getName() const override;
        };
    }
}