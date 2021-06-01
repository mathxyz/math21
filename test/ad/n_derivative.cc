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

#include "n_derivative.h"

namespace math21 {
    namespace ad {
        // tanh(x)
        void test_n_derivative_tanh_dbr() {
//            ad::Variable::setRequestedAbstractLevel(1);
            ad::Variable::setRequestedAbstractLevel(0);
            VariableMap &data = ad_get_data();

            auto x = ad_create_point_var("x");
            Variable &vx = ad_get_variable(x);
            vx.setType(variable_type_input);
            vx.getValue().setSize(1);
            vx.getValue() = 1.0;
            vx.synchronizeToZero(data);

            auto y = tanh(x);
            y.log("0", 15);
            NumN n = 1;
//            NumN n = 2;
//            NumN n = 3;
//            NumN n = 6;
            for (NumN i = 1; i <= n; ++i) {
                y = grad(x, y);
                if (y.id == 0) {
                    m21log("f=0, so stop computing f'");
                    m21log("i", i);
                    break;
                }
                y.log(math21_string_to_string(i).c_str(), 15);
            }

            // finite differences
            auto y_fd = (tanh(1.0001) - tanh(0.9999)) / 0.0002;
            y_fd.log("y_fd", 15);
        }

        void test_n_derivative_mat_mul() {
            VariableMap &data = ad_get_data();

            auto w = ad_create_point_var("W");
            Variable &vw = ad_get_variable(w);
            vw.setType(variable_type_input);
            vw.getValue().setSize(2, 2);
            vw.getValue().letters();

            auto x = ad_create_point_var("X");
            Variable &vx = ad_get_variable(x);
            vx.setType(variable_type_input);
            vx.getValue().setSize(2, 2);
            vx.getValue() = 5, 6, 7, 8;

            auto y = ad_mat_mul(w, x);
            y.log("0-th order", 15);
            NumN n = 1;
//    NumN n = 2;
            for (NumN i = 1; i <= n; ++i) {
//        y = ad_grad(w, y);
                y = ad_jacobian(w, y);
                if (y.id == 0) {
                    m21log("f=0, so stop computing f'");
                    m21log("i order", i);
                    break;
                }
                y.log((math21_string_to_string(i)+"-th order").c_str(), 15);
            }
        }

        void test_n_derivative_all() {
            test_n_derivative_tanh_dbr();
            test_n_derivative_mat_mul();
        }
    }
}