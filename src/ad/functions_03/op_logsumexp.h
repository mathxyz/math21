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
        // y = f(x), L = L(y), L in R
        struct op_logsumexp : public op_m_to_n {
        private:
            VecN axes;
            NumB isKeepingDims; // only applies to y, not dL/dy
        public:
            explicit op_logsumexp(const VecN &axes = VecN(), NumB isKeepingDims = 0);

            ~op_logsumexp() override = default;

            NumN df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const override;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            void evaluate_at_vec(const VecR &x, VecR &y) const override;

            Function *clone() const override {
                auto *f = new op_logsumexp(axes, isKeepingDims);
                return f;
            }

            const char *getName() const override {
                return "op_logsumexp";
            }
        };
    }
}