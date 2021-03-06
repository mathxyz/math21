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
        struct op_mat_mul : public Function {
        private:
        public:
            op_mat_mul();

            virtual ~op_mat_mul();

            NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            NumN evaluate(const Set &X, VariableMap &data) override;

            void cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void f(const Set &X, Set &output, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            void compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative) override;

            void setSize(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override;

            const char *getName() const override;

        };

        struct op_mat_dmul : public Function {
        private:
            NumN _varId; // 1 for W, 2 for X
        public:
            op_mat_dmul(NumN varId);

            virtual ~op_mat_dmul();

            void cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void f(const Set &X, Set &output, VariableMap &data) override;

            // dot product
            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            void setSize(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override;

            const char *getName() const override;

        };
    }
}