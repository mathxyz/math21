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
    struct mlfunction_conv_detail {
        m21variable K_wrapper;
        m21variable dK_wrapper;
        m21variable b_wrapper; // empty when batch normalization.
        m21variable db_wrapper;
        m21variable y_wrapper;
        m21variable dy_wrapper;
        _Map<std::string, m21variable *> vars;

        mlfunction_conv_detail() {
            K_wrapper.setName("K");
            dK_wrapper.setName("dK");
            b_wrapper.setName("b");
            db_wrapper.setName("db");
            y_wrapper.setName("y");
            dy_wrapper.setName("dy");
            vars.add(K_wrapper.getName(), &K_wrapper);
            vars.add(dK_wrapper.getName(), &dK_wrapper);
            vars.add(b_wrapper.getName(), &b_wrapper);
            vars.add(db_wrapper.getName(), &db_wrapper);
            vars.add(y_wrapper.getName(), &y_wrapper);
            vars.add(dy_wrapper.getName(), &dy_wrapper);
        }

        const _Map<std::string, m21variable *> &getVars() const {
            return vars;
        }
    };
}
