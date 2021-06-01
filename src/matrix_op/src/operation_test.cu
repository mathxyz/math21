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

#include "inner_cc.h"
#include "../operations.h"

namespace math21 {

    //////////////////////// test

    void math21_cuda_test_02() {
        NumN m, r, n;

        m = 1 << 9;
        r = 1 << 9;
        n = 1 << 9;

        TenR A(m, r);
        TenR B(r, n);
        TenR C(m, n);
        TenR C2(m, n);

        DefaultRandomEngine engine(21);
        RanUniform ran(engine);
        ran.set(0, 1000);
        math21_random_draw(A, ran);
        math21_random_draw(B, ran);

        NumR gpu_elapsed_time_ms, cpu_elapsed_time_ms;

        timer t;

        t.start();
        math21_operator_multiply(1, A, B, C);
        t.end();
        printf("Time elapsed %f ms.\n\n", gpu_elapsed_time_ms = t.time());

        t.start();
        math21_c_matrix_multiply(1, A, B, C2);
        t.end();
        printf("Time elapsed %f ms.\n\n", cpu_elapsed_time_ms = t.time());

        MATH21_ASSERT(math21_operator_container_isEqual(C, C2, 1e-6));

        // roughly compute speedup
        printf("speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
}