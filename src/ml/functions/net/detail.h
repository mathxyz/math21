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

#include "inner_cc.h"

namespace math21 {
    struct mlfunction_net_detail {
        m21variable data_x_wrapper;
        m21variable data_y_wrapper;
        _Map<std::string, m21variable *> vars;

        mlfunction_net_detail() {
            data_x_wrapper.setName("data_x");
            data_y_wrapper.setName("data_y");
            vars.add(data_x_wrapper.getName(), &data_x_wrapper);
            vars.add(data_y_wrapper.getName(), &data_y_wrapper);
        }

        const _Map<std::string, m21variable *> &getVars() const {
            return vars;
        }
    };
}
