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
    class Functional : public think::Operator {
    private:
        ArrayN p;
        VecN v;
    public:
        Functional() {}

        virtual ~Functional() {}

        //y=T(x)
//        virtual NumR valueAt(const VecR &x)=0;
        virtual NumR valueAt(const TenR &x)=0;

        //Todo: think if can add const.
        virtual NumN getXDim()=0;

        virtual const TenR &derivativeValueAt(const TenR &x)=0;

//        virtual NumR valueAt(const think::Point &x) {
//            MATH21_ASSERT_NOT_CALL(0, "You must overwrite to use");
//            return 0;
//        }

        virtual const ArrayN &get_x_shape() const {
            MATH21_ASSERT_NOT_CALL(0, "You must overwrite to use");
            return p;
        }

        //deprecate
//        virtual const VecN &derivativeValueAt(const think::Point &x) {
//            MATH21_ASSERT_NOT_CALL(0, "You must overwrite to use");
//            return v;
//        }
    };
}