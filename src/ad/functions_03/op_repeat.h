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
        struct op_repeat : public op_m_to_n {
        public:
            op_repeat() = default;

            ~op_repeat() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            void evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const override;

            Function *clone() const override {
                auto *f = new op_repeat();
                return f;
            }

            const char *getName() const override {
                return "op_repeat";
            }
        };

        struct op_undo_repeat_sum : public op_m_to_n {
        public:
            op_undo_repeat_sum() = default;

            ~op_undo_repeat_sum() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            void evaluate_at_vec(const VecR &x1, const VecR &x2, const VecR &x3, VecR &y) const override;

            Function *clone() const override {
                auto *f = new op_undo_repeat_sum();
                return f;
            }

            const char *getName() const override {
                return "op_undo_repeat_sum";
            }
        };
    }
}