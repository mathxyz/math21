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
        // todo: merge with op_num_binary
        // changed from op_num_unary
        struct op_ele_unary : public Function {
        private:
        public:
            op_ele_unary();

            virtual ~op_ele_unary();

            virtual NumN df_vjp(const Set &X, NumN x, NumN y, VariableMap &data) const;

            virtual NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const override;

            virtual NumN evaluate(const Set &X, VariableMap &data) override;

            void cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void f(const Set &X, Set &output, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

//            virtual NumR evaluate_at_num(NumR x) const = 0;

            virtual void evaluate_at_tensor(const VecR &x, VecR &y) const = 0;
        };
    }
}