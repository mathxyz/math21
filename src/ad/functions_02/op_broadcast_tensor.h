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
        struct op_broadcast_tensor : public Function {
        public:
            op_broadcast_tensor() = default;

            ~op_broadcast_tensor() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_broadcast_tensor();
                return f;
            }

            const char *getName() const override {
                return "op_broadcast_tensor";
            }
        };

        struct op_create : public Function {
        public:
            op_create() = default;

            ~op_create() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_create();
                return f;
            }

            const char *getName() const override {
                return "op_create";
            }
        };
    }
}