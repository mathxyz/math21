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
        struct op_num_assign : public Function {
        private:
        public:
            op_num_assign();

            virtual ~op_num_assign();

            void df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const;

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