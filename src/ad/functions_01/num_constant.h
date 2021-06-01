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
        // constant variable
        // It is treated as variable and has derivative, but can't propagate back further.
        // default value 0
        struct op_num_constant : public Function {
        private:
        public:
            op_num_constant();

            virtual ~op_num_constant();

            void df(const Set &X, NumN x, NumN y, Set &output, VariableMap &data) const;

            void cr(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const override;

            void f(NumN &y, VariableMap &data);

            void f(const Set &X, Set &output, VariableMap &data) override;

            void fv(const Set &X, const Set &Y, VariableMap &data) const override;

            void setSize(const Set &X, const Set &Y, VariableMap &data) const override;

            Function *clone() const override;

            const char *getName() const override;
        };
    }
}