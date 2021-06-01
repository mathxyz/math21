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
        struct op_get_shape : public Function {
        public:
            op_get_shape() = default;

            ~op_get_shape() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            NumN evaluate(const Set &X, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_get_shape();
                return f;
            }

            const char *getName() const override {
                return "op_get_shape";
            }
        };

        struct op_get_size : public Function {
        public:
            op_get_size() = default;

            ~op_get_size() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            NumN evaluate(const Set &X, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_get_size();
                return f;
            }

            const char *getName() const override {
                return "op_get_size";
            }
        };

        struct op_get_shrink_shape_keeping_dim : public Function {
        public:
            op_get_shrink_shape_keeping_dim() = default;

            ~op_get_shrink_shape_keeping_dim() override = default;

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            NumN evaluate(const Set &X, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override {
                auto *f = new op_get_shrink_shape_keeping_dim();
                return f;
            }

            const char *getName() const override {
                return "op_get_shrink_shape_keeping_dim";
            }
        };
    }
}