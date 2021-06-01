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
        void ad_global_create(VariableMap &data);

        void ad_global_destroy();

        Function *ad_global_get_op_do_nothing();

        Function *ad_global_get_op_num_input();

        Function *ad_global_get_op_num_constant();

        NumN ad_global_get_constant_0();

        NumN ad_global_get_constant_1();

        NumN ad_global_get_constant_m1();
    }
}