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
    class f_example_2 : public Functional {
    private:
        MatR A;
        VecR d;
        VecR x0;
        VecR B;
        MatR output, output2, output3, output4; // tmp
    public:
        f_example_2();

        virtual ~f_example_2() {}

        NumR valueAt(const VecR &x) override;

        NumN getXDim() override;

        const VecR &getX0();

        const VecR &derivativeValueAt(const VecR &x) override;
//        VecR map(const VecR &x){
//            return x;
//        }
    };

}