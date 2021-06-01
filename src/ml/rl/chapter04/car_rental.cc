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
#include "car_rental.h"

namespace math21 {

    namespace rl {
        namespace car_rental {
            NumN M1 = 20;
            NumN M2 = 20;
            NumN Mm = 5;
            NumN lambda11, lambda12, lambda21, lambda22;
            NumR gamma = 0.9;
            NumR R_rent = 10;
            NumR c_move = 2;
            VecZ A;
            NumN M_poisson = 10;
            NumR delta_M = 1e-4;
            Dict<NumN, NumR> poisson_cache;

            // Note: lambda < 10
            NumR poisson_probability(NumN n, NumN lambda) {
                NumN key = n * 10 + lambda + 1;
                NumN index = poisson_cache.has(key);
                if (index == 0) {
                    NumR value = math21_pr_poisson_probability(n, lambda);
                    poisson_cache.add(key, value);
                    return value;
                } else {
                    return poisson_cache.valueAtIndex(index);
                }
            }

            // Bellman equation for q(s, a)
            NumR Bellman_for_q_s_a(const VecN &s, NumZ a,
                                   const IndexFunctional<NumR> &V,
                                   NumB is_const_returned_cars) {
                NumR EG = 0;
                EG = EG - xjabs(a) * c_move;
                NumN n1, n2;
                n1 = xjmin(s(1) - a, M1);
                n2 = xjmin(s(2) + a, M2);
                NumN req1, req2;
                for (req1 = 0; req1 <= M_poisson; ++req1) {
                    for (req2 = 0; req2 <= M_poisson; ++req2) {
                        NumR p;
                        p = poisson_probability(req1, lambda11)
                            * poisson_probability(req2, lambda21);
                        NumN n1_prime, n2_prime;
                        n1_prime = n1;
                        n2_prime = n2;
                        NumN n1v, n2v;
                        n1v = xjmin(n1_prime, req1);
                        n2v = xjmin(n2_prime, req2);
                        NumR r;
                        r = (n1v + n2v) * R_rent;
                        n1_prime = n1_prime - n1v;
                        n2_prime = n2_prime - n2v;
                        if (is_const_returned_cars) {
                            NumN ret1, ret2;
                            ret1 = lambda12;
                            ret2 = lambda22;
                            n1_prime = xjmin(n1_prime + ret1, M1);
                            n2_prime = xjmin(n2_prime + ret2, M2);
                            VecN s_prime(2);
                            s_prime = n1_prime, n2_prime;
                            EG = EG + p * (r + gamma * V(s_prime));
                        } else {
                            NumR EG_prime = 0;
                            NumN ret1, ret2;
                            VecN s_prime(2);
                            for (ret1 = 0; ret1 <= M_poisson; ++ret1) {
                                for (ret2 = 0; ret2 <= M_poisson; ++ret2) {
                                    NumR p_prime;
                                    p_prime = poisson_probability(ret1, lambda12)
                                              * poisson_probability(ret2, lambda22);
                                    n1_prime = xjmin(n1_prime + ret1, M1);
                                    n2_prime = xjmin(n2_prime + ret2, M2);
                                    s_prime = n1_prime, n2_prime;
                                    EG_prime = EG_prime + p_prime * V(s_prime);
                                }
                            }
                            EG = EG + p * (r + gamma * EG_prime);
                        }
                    }
                }
                return EG;
            }

            void figure_4_2(ImageDraw &f, NumB is_const_returned_cars = 1) {
                IndexFunctional<NumR> V;
                V.setSize(M1 + 1, M2 + 1);
                V.setStartIndex(0, 0);
                V.getTensor() = 0;
                IndexFunctional<NumZ> pi;
                pi.setSize(M1 + 1, M2 + 1);
                pi.setStartIndex(0, 0);
                pi.getTensor() = 0;
                NumN k = 0;
                TenR pi_draw;
                pi_draw.setSize(pi.getTensor().shape());
                VecN s(2);
                VecR qsa(A.size());
                TenZ pi_draw_log;
                pi_draw_log.setSize(pi.getTensor().shape());

                NumR t = math21_time_getticks();
                while (1) {
                    math21_operator_matrix_reverse_x_axis(pi.getTensor(), pi_draw_log);
                    std::string name = "policy " + math21_string_to_string(k);
                    pi_draw_log.log(name.c_str());

                    math21_operator_tensor_assign_elementwise(pi_draw, pi.getTensor());
                    f.setDir("figure_4_2");
                    f.draw(pi_draw, name.c_str());

                    // policy evaluation
                    NumN k2 = 0;
                    while (1) {
                        NumR delta = 0;
                        for (NumN i = 0; i <= M1; ++i) {
                            for (NumN j = 0; j <= M2; ++j) {
                                s = i, j;
                                NumR v = V(s);
                                V(s) = Bellman_for_q_s_a(s, pi(s), V, is_const_returned_cars);
                                delta = xjmax(delta, xjabs(v - V(s)));
                            }
                        }
                        if (delta < delta_M) {
                            std::cout << "delta = " << delta << " in iteration " << k2 << "\n";
                            break;
                        }
                        if (k2 % 2 == 0) {
                            std::cout << "delta = " << delta << " in iteration " << k2 << "\n";
                        }
                        k2++;
                    }

                    // policy improvement
                    NumB is_policy_stable = 1;
                    for (NumN i = 0; i <= M1; ++i) {
                        for (NumN j = 0; j <= M2; ++j) {
                            s = i, j;
                            NumZ a_old = pi(s);
                            for (NumN ia = 1; ia <= A.size(); ++ia) {
                                NumZ a = A(ia);
                                if (-(NumZ) s(2) <= a && a <= (NumZ) s(1)) {
                                    qsa(ia) = Bellman_for_q_s_a(s, a, V, is_const_returned_cars);
                                } else {
                                    qsa(ia) = MATH21_MIN;
                                }
                            }
                            pi(s) = A(math21_operator_container_argmax(qsa));
                            if (is_policy_stable && a_old != pi(s)) {
                                is_policy_stable = 0;
                            }
                        }
                    }
                    if (is_policy_stable) {
                        V.getTensor().log("optimal_value");
                        f.draw(V.getTensor(), "optimal_value");
                        break;
                    }
                    k = k + 1;
                }

                t = math21_time_getticks() - t;
                std::cout << "Optimal policy is reached after "
                          << k << " iterations in " << t << " seconds\n";

            }

            void init() {
                lambda11 = 3, lambda12 = 3, lambda21 = 4, lambda22 = 2;

                M_poisson = 8;
                delta_M = 1e-1;

                A.setSize(2 * Mm + 1);
                math21_operator_container_letters(A, -(NumZ) Mm);
            }

            void test(ImageDraw &f) {
                init();
//                figure_4_2(f, 1);
                figure_4_2(f, 0);
            }
        }
    }

    void math21_rl_chapter04_car_rental(ImageDraw &f) {
        m21log("car rental");
        rl::car_rental::test(f);
    }
}