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

#include "../inner.h"

namespace math21 {
    namespace think {
        /*
         * y=T(x)
         *
         * */
        class Operator {
        public:
            Operator() {}

            virtual ~Operator() {}

            //y=T(x)
//            virtual think::Point map(think::Point &x)=0;
        };

        // random number generator
        struct RandomEngine {
        public:
            RandomEngine() {
            }

            virtual ~RandomEngine() {
            }

            // [0.0, 1.0]
            virtual NumR draw_0_1() = 0;

            virtual NumN draw_NumN() = 0;
        };

        // distribution
        struct Random {
        public:
            Random() {
            }

            virtual ~Random() {
            }

            virtual void draw(NumR &x) = 0;

            virtual void draw(NumN &x) = 0;

            virtual void draw(NumZ &x) = 0;
        };

    }
}