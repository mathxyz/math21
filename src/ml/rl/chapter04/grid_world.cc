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
        namespace chapter04_grid_world {
            NumN X_MAX = 4;
            NumN Y_MAX = 4;
            // A and B are terminal states.
            VecN A_POS;
            VecN B_POS;
            NumR gamma = 0.9;
            Seqce<VecZ> As; // left, up, right, down
            NumR ACTION_PROB = 0.25;

            void init() {
                gamma = 1.0;

                A_POS.setSize(2);
                A_POS = 1, Y_MAX;
                B_POS.setSize(2);
                B_POS = X_MAX, 1;

                As.setSize(4);
                math21_operator_container_element_setSize(As, 2);
                As.at(1) = -1, 0;
                As.at(2) = 0, 1;
                As.at(3) = 1, 0;
                As.at(4) = 0, -1;
            }

            NumB isTerminal(const VecN &s) {
                if (math21_operator_container_isEqual(s, A_POS)
                    || math21_operator_container_isEqual(s, B_POS)) {
                    return 1;
                } else {
                    return 0;
                }
            }

            void step(const VecN &s, const VecZ &a, VecN &s_prime, NumR &r) {
                MATH21_ASSERT(s_prime.isSameSize(s.shape()))

                if (isTerminal(s)) {
                    s_prime.assign(s);
                    r = 0;
                    return;
                }

                math21_operator_container_addToC(s, a, s_prime);
                NumN x, y;
                x = s_prime(1);
                y = s_prime(2);
                if (x < 1 || x > X_MAX || y < 1 || y > Y_MAX) {
                    s_prime.assign(s);
                }
                r = -1;
            }

            // iteration: k;
            void compute_state_value(MatR &v, NumN &k, NumB isInPlace) {
                v = 0;
                k = 0;
                MatR *p_v_new;
                MatR *p_v_old;
                MatR v_new_tmp;
                MatR v_old_tmp;
                if (isInPlace) {
                    p_v_new = &v;
                    v_old_tmp.copyFrom(v);
                    p_v_old = &v_old_tmp;
                } else {
                    v_new_tmp.copyFrom(v);
                    p_v_new = &v_new_tmp;
                    p_v_old = &v;
                }
                MatR &v_new = *p_v_new;
                MatR &v_old = *p_v_old;

                MatR v_draw;
                MatR error(X_MAX, Y_MAX);
                NumN i, j, ia;
                NumN As_size = As.size();
                VecN s(2);
                VecN s_prime(2);
                NumR r;
                NumR pi_a_s;
                NumR delta;
                while (1) {
                    if (isInPlace) {
                        v_old.assign(v);
                    }
                    for (i = 1; i <= X_MAX; ++i) {
                        for (j = 1; j <= Y_MAX; ++j) {
                            s = i, j;
                            NumR value = 0;
                            for (ia = 1; ia <= As_size; ++ia) {
                                const VecZ &a = As(ia);
                                step(s, a, s_prime, r);

                                pi_a_s = ACTION_PROB;
                                value = value + pi_a_s * (r + gamma * v(s_prime));
                            }
                            v_new(s) = value;
                        }
                    }

                    math21_operator_container_subtract_to_C(v_new, v_old, error);
                    math21_operator_container_abs_to(error);
                    delta = math21_operator_container_max(error);
                    if (!isInPlace) {
                        v.assign(v_new);
                    }
                    if (delta < 1e-4) {
                        break;
                    }
                    ++k;
                }
            }

            void figure_4_1() {
                MatR v(X_MAX, Y_MAX);
                MatR v_draw;
                NumN asycn_iteration, sync_iteration;

                compute_state_value(v, asycn_iteration, 0);
                math21_operator_matrix_axis_to_image(v, v_draw);
                v_draw.log("v in-place");

                compute_state_value(v, sync_iteration, 1);
                math21_operator_matrix_axis_to_image(v, v_draw);
                v_draw.log("v synchronous");

                std::cout << "In-place: " << asycn_iteration << " iterations\n";
                std::cout << "Synchronous: " << sync_iteration << " iterations\n";
            }


            void test() {
                init();
                figure_4_1();
            }
        }
    }

    void math21_rl_chapter04_grid_world() {
        m21log("grid world");
        rl::chapter04_grid_world::test();
    }
}