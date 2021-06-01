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
    /*
     * y=T(x)
     *
     * */
    class Function : public think::Operator {
    public:
        Function() {}

        //must be virtual.
        virtual ~Function() {}

        //y=f(x)
        virtual NumR valueAt(const NumR &x)=0;

        virtual NumR derivativeValueAt(const NumR &x)=0;

        virtual NumR derivativeValue_using_y(const NumR &y){
            MATH21_ASSERT_NOT_CALL(0, "You must overwrite to use");
            return 1;
        }

        // y = f(x)
//        virtual NumR derivativeValueAt(const NumR &x, const NumR &y) {
//            MATH21_ASSERT_NOT_CALL(0, "You must overwrite to use");
//            return 1;
//        }
    };
}