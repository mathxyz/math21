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
    class polynomial : public Functional {
    private:
        VecR A;
        VecR x0;
    public:
        polynomial() : Functional() {
            A.setSize(1);
            x0.setSize(1);
//        x0 = 0.8;
            x0 = 300;
        }

        virtual ~polynomial() {}

        NumR valueAt(const VecR &x) override;

        NumN getXDim() override;

        const VecR &getX0();

        const VecR &derivativeValueAt(const VecR &x) override;
//        VecR map(const VecR &x){
//            return x;
//        }
    };
}