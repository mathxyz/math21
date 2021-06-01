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

#include "op_do_nothing.h"
#include "num_input.h"
#include "num_constant.h"
#include "ad_global.h"

namespace math21 {
    namespace ad {
        struct ad_global {
            VariableMap &data;
            op_do_nothing *do_nothing;
            op_num_input *num_input;
            op_num_constant *num_constant;
        public:
            ad_global(VariableMap &data) : data(data) {
                do_nothing = new op_do_nothing();
                do_nothing->setGlobalFlag(1);
                num_input = new op_num_input();
                num_input->setGlobalFlag(1);
                num_constant = new op_num_constant();
                num_constant->setGlobalFlag(1);
            }

            virtual ~ad_global() {
                delete do_nothing;
                delete num_input;
                delete num_constant;
            }
        };

        ad_global *global = 0;

        void ad_global_create(VariableMap &data) {
            MATH21_ASSERT(global == 0)
            global = new ad_global(data);
        }

        // It had better be called at last.
        void ad_global_destroy() {
            if (global) {
                delete global;
                global = 0;
            }
        }

        Function *ad_global_get_op_do_nothing() {
            MATH21_ASSERT(global != 0)
            return global->do_nothing;
        }

        Function *ad_global_get_op_num_input() {
            MATH21_ASSERT(global != 0)
            return global->num_input;
        }

        Function *ad_global_get_op_num_constant() {
            MATH21_ASSERT(global != 0)
            return global->num_constant;
        }

        // todo: error, should make it different.
        NumN ad_global_get_constant_0() {
            MATH21_ASSERT(global != 0)
            return global->data.get_constant_0();
        }

        NumN ad_global_get_constant_1() {
            MATH21_ASSERT(global != 0)
            return global->data.get_constant_1();
        }

        // -1
        NumN ad_global_get_constant_m1() {
            MATH21_ASSERT(global != 0)
            return global->data.get_constant_m1();
        }
    }
}