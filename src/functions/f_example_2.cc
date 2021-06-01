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

#include "f_example_2.h"

namespace math21 {


    /*
     * f(x) = 1 / 2 * x.trans() * A * x + d.trans()*x
     * x* = (0, 0)
     * */
    f_example_2::f_example_2() : Functional() {
        A.setSize(2, 2);
        A = 2, 1, 1, 2;
        d.setSize(2);
        d = 0;
        x0.setSize(2);
//        x0 = 0.8, -0.25;
        x0 = 300, 200;
        B.setSize(2);
        B = 0;
    }

    // (1 / 2.0 * x.trans() * A * x + d.trans() * x)(1, 1);
    NumR f_example_2::valueAt(const VecR &x) {
        math21_operator_trans_multiply(1 / 2.0, x, A, output);
        math21_operator_multiply(1, output, x, output2);
        math21_operator_trans_multiply(1, d, x, output3);
        math21_operator_linear_to_B(1, output2, 1, output3);
        return output3(1, 1);
    }

    NumN f_example_2::getXDim() {
        return 2;
    }

    const VecR &f_example_2::getX0() {
        return x0;
    }

    //(x.trans() * A + d.trans()).trans();
    const VecR &f_example_2::derivativeValueAt(const VecR &x) {
//        m21log("optimalaa",valueAt(x0));
        math21_operator_trans_multiply(1, A, x, output4);
        math21_operator_linear_to_A(1, output4, 1, d);
        B(1) = output4(1, 1);
        B(2) = output4(2, 1);
        return B;
    }


}