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

#include "inner.h"
#include "grid_world.h"

namespace math21 {
    namespace rl {
        namespace grid_world {
            NumN X_MAX = 5;
            NumN Y_MAX = 5;
            VecN A_POS;
            VecN A_prime_POS;
            VecN B_POS;
            VecN B_prime_POS;
            NumR R_A = 10;
            NumR R_B = 5;
            NumR gamma = 0.9;
            Seqce<VecZ> As; // left, up, right, down
            NumR ACTION_PROB = 0.25;

            void init() {
                A_POS.setSize(2);
                A_POS = 2, 5;
                A_prime_POS.setSize(2);
                A_prime_POS = 2, 1;
                B_POS.setSize(2);
                B_POS = 4, 5;
                B_prime_POS.setSize(2);
                B_prime_POS = 4, 3;

                As.setSize(4);
                math21_operator_container_element_setSize(As, 2);
                As.at(1) = -1, 0;
                As.at(2) = 0, 1;
                As.at(3) = 1, 0;
                As.at(4) = 0, -1;
            }

            void step(const VecN &s, const VecZ &a, VecN &s_prime, NumR &r) {
                MATH21_ASSERT(s_prime.isSameSize(s.shape()))

                if (math21_operator_isEqual(s, A_POS)) {
                    s_prime.assign(A_prime_POS);
                    r = R_A;
                    return;
                }
                if (math21_operator_isEqual(s, B_POS)) {
                    s_prime.assign(B_prime_POS);
                    r = R_B;
                    return;
                }
                math21_operator_container_addToC(s, a, s_prime);
                NumN x, y;
                x = s_prime(1);
                y = s_prime(2);
                if (x < 1 || x > X_MAX || y < 1 || y > Y_MAX) {
                    s_prime.assign(s);
                    r = -1;
                } else {
                    r = 0;
                }
            }

            void figure_3_2() {
                MatR v(X_MAX, Y_MAX);
                MatR v_draw;
                v = 0;
                MatR v_prime(X_MAX, Y_MAX);
                MatR error(X_MAX, Y_MAX);
                NumN i, j, k;
                NumN As_size = As.size();
                VecN s(2);
                VecN s_prime(2);
                NumR r;
                NumR pi_a_s;

                while (1) {
                    v_prime = 0;
                    for (i = 1; i <= X_MAX; ++i) {
                        for (j = 1; j <= Y_MAX; ++j) {
                            s = i, j;
                            for (k = 1; k <= As_size; ++k) {
                                const VecZ &a = As(k);
                                step(s, a, s_prime, r);
                                pi_a_s = ACTION_PROB;
                                v_prime(s) += pi_a_s * (r + gamma * v(s_prime));
                            }
                        }
                    }

                    math21_operator_linear(1, v, -1, v_prime, error);
                    if(math21_operator_container_norm(error, 1) < 1e-4 ){
                        math21_operator_matrix_axis_to_image(v, v_draw);
                        v_draw.log("v");
                        return;
                    }
                    v.assign(v_prime);
                }
            }

            void figure_3_5() {
                MatR v(X_MAX, Y_MAX);
                MatR v_draw;
                v = 0;
                MatR v_prime(X_MAX, Y_MAX);
                MatR error(X_MAX, Y_MAX);
                NumN i, j, k;
                NumN As_size = As.size();
                VecR vs(As_size);
                VecN s(2);
                VecN s_prime(2);
                NumR r;
                while (1) {
                    v_prime = 0;
                    for (i = 1; i <= X_MAX; ++i) {
                        for (j = 1; j <= Y_MAX; ++j) {
                            s = i, j;
                            vs = 0;
                            for (k = 1; k <= As_size; ++k) {
                                const VecZ &a = As(k);
                                step(s, a, s_prime, r);
                                vs(k) = r + gamma * v(s_prime);
                            }
                            v_prime(s) = math21_operator_container_max(vs);
                        }
                    }

                    math21_operator_linear(1, v, -1, v_prime, error);
                    if(math21_operator_container_norm(error, 1) < 1e-4 ){
                        math21_operator_matrix_axis_to_image(v, v_draw);
                        v_draw.log("v");
                        return;
                    }
                    v.assign(v_prime);
                }
            }



            void test() {
                init();
                figure_3_2();
                figure_3_5();
            }
        }
    }

    void math21_rl_chapter03_grid_world() {
        m21log("grid world");
        rl::grid_world::test();
    }
}