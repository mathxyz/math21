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
#include "car_rental_synchronous.h"

namespace math21 {

    namespace rl {
        namespace car_rental_synchronous {
            // Todo: run when is_const_returned_cars = 0


            // Note: all these variables must be constant
            // when compute because of parallel processing.
            NumN M1 = 20;
            NumN M2 = 20;
            NumN Mm = 5;
            NumN lambda11, lambda12, lambda21, lambda22;
            NumR gamma = 0.9;
            NumR R_rent = 10;
            NumR c_move = 2;
            NumR c_park = 4;
            NumN M_poisson = 8;
            NumR delta_M = 1e-1;
            NumB is_const_returned_cars = 1;
            NumB is_solve_4_5 = 0;
            Dict<NumN, NumR> poisson_cache;

            VecZ A;
            IndexFunctional<NumR> V;
            IndexFunctional<NumZ> pi;

            // we must run this function once
            // because we are using parallel processing which prohibits global variables.
            // Note: lambda < 10
            void poisson_probability_init(NumN n, NumN lambda) {
                NumN key = n * 10 + lambda + 1;
                NumN index = poisson_cache.has(key);
                if (index == 0) {
                    NumR value = xjexp(-(NumZ) lambda) * xjpow(lambda, n) / xjfactorial(n);
                    poisson_cache.add(key, value);
                }
            }

            // use constant because of possible parallel computing.
            NumR poisson_probability(NumN n, NumN lambda) {
                const Dict<NumN, NumR> &m_poisson_cache = poisson_cache;
                NumN key = n * 10 + lambda + 1;
                NumN index = m_poisson_cache.has(key);
                if (index != 0) {
                    return m_poisson_cache.valueAtIndex(index);
                } else {
                    MATH21_ASSERT(0, "run init first!")
                    return 0;
                }
            }

            void set_poisson_cache() {
                VecN lambdas(4);
                lambdas = lambda11, lambda12, lambda21, lambda22;
                for (NumN i = 1; i <= lambdas.size(); ++i) {
                    NumN lam = lambdas(i);
                    for (NumN n = 0; n <= M_poisson; ++n) {
                        poisson_probability_init(n, lam);
                    }
                }
            }

            void init() {
                lambda11 = 3, lambda12 = 3, lambda21 = 4, lambda22 = 2;

                M_poisson = 8;
                delta_M = 1e-1;

//                is_const_returned_cars = 1;
                is_const_returned_cars = 0;
                is_solve_4_5 = 0;
//                is_solve_4_5 = 1;

                A.setSize(2 * Mm + 1);
                math21_operator_container_letters(A, -(NumZ) Mm);

                V.setSize(M1 + 1, M2 + 1);
                V.setStartIndex(0, 0);
                pi.setSize(M1 + 1, M2 + 1);
                pi.setStartIndex(0, 0);

                set_poisson_cache();
            }

            // Bellman equation for q(s, a)
            NumR Bellman_for_q_s_a(const VecN &s, NumZ a,
                                   const IndexFunctional<NumR> &V,
                                   NumB is_const_returned_cars) {
                NumR EG = 0;
                if (is_solve_4_5) {
                    if (a > 0) {
                        EG = EG - (a - 1) * c_move;
                    } else {
                        EG = EG - xjabs(a) * c_move;
                    }
                } else {
                    EG = EG - xjabs(a) * c_move;
                }
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
                        if (is_solve_4_5) {
                            if (n1_prime >= 10) {
                                r = r - c_park;
                            }
                            if (n2_prime >= 10) {
                                r = r - c_park;
                            }
                        }
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

            NumR v_pi_s_pe(const IndexFunctional<NumZ> &pi,
                           const IndexFunctional<NumR> &V,
                           const VecN &s) {
                NumZ a = pi(s);
                return Bellman_for_q_s_a(s, a, V, is_const_returned_cars);
            }

            NumR q_pi_s_a_pi(const IndexFunctional<NumR> &V,
                             const VecN &s,
                             NumZ a) {
                NumR EG;
                if (-(NumZ) s(2) <= a && a <= (NumZ) s(1)) {
                    EG = Bellman_for_q_s_a(s, a, V, is_const_returned_cars);
                } else {
                    EG = MATH21_MIN;
                }
                return EG;
            }

            // out-place
            void policy_evaluation(IndexFunctional<NumR> &V,
                                   const IndexFunctional<NumZ> &pi) {
                IndexFunctional<NumR> V_new;
                V_new.setSize(M1 + 1, M2 + 1);
                V_new.setStartIndex(0, 0);
                MatR error(M1 + 1, M2 + 1);
                while (1) {
#pragma omp parallel for collapse(2)
                    for (NumN i = 0; i <= M1; ++i) {
                        for (NumN j = 0; j <= M2; ++j) {
                            VecN s(2);
                            s = i, j;
                            V_new(s) = v_pi_s_pe(pi, V, s);
                        }
                    }
                    math21_operator_container_subtract_to_C(V.getTensor(), V_new.getTensor(), error);
                    if (math21_operator_container_norm(error, 1) < delta_M) {
                        V.getTensor().assign(V_new.getTensor());
                        return;
                    }
                    V.getTensor().assign(V_new.getTensor());
                }
            }

            NumB policy_improvement(const VecZ &A, const IndexFunctional<NumR> &V,
                                    IndexFunctional<NumZ> &pi) {

                NumN k = 0;
                IndexFunctional<NumZ> pi_new;
                m21warn("Type of pi_new can be changed to tensor");
                pi_new.setSize(M1 + 1, M2 + 1);
                IndexFunctional<NumR> qsa;
                qsa.setSize(M1 + 1, M2 + 1, A.size());
                qsa.setStartIndex(0, 0, -(NumZ) Mm);
                NumN A_size = A.size();
                NumN i, j, ia;
#pragma omp parallel for private(i, j, ia) collapse(3)
                for (i = 0; i <= M1; ++i) {
                    for (j = 0; j <= M2; ++j) {
                        for (ia = 1; ia <= A_size; ++ia) {
                            VecN s(2);
                            s = i, j;
                            NumZ a = A(ia);
                            qsa(i, j, a) = q_pi_s_a_pi(V, s, a);
                        }
                    }
                }

                TenR B;
                VecN index(qsa.getTensor().dims());
                index = 0, 0, 1;

                TensorFunction_argmax f_arg_max;
                math21_operator_tensor_f_shrink(qsa.getTensor(), pi_new.getTensor(), index, f_arg_max);
                NumZ tensor_offset = -qsa.getTensor_index_offset(3);
                math21_operator_container_addTo(tensor_offset, pi_new.getTensor());

                if (math21_operator_container_isEqual(pi.getTensor(), pi_new.getTensor()) == 0) {
                    pi.getTensor().assign(pi_new.getTensor());
                    return 1;
                }
                return 0;
            }

            void policy_iteration(ImageDraw &f) {
                V.getTensor() = 0;
                pi.getTensor() = 0;

                TenR pi_draw;
                pi_draw.setSize(pi.getTensor().shape());

                TenZ pi_draw_log;
                pi_draw_log.setSize(pi.getTensor().shape());

                NumN k = 0;
                NumB is_policy_changed;
                NumR t = math21_time_getticks();
                while (1) {
                    math21_operator_matrix_reverse_x_axis(pi.getTensor(), pi_draw_log);
                    std::string name = "policy " + math21_string_to_string(k);
                    pi_draw_log.log(name.c_str());

                    math21_operator_tensor_assign_elementwise(pi_draw, pi.getTensor());
                    f.setDir("figure_4_2_synchronous");
                    f.draw(pi_draw, name.c_str());

                    policy_evaluation(V, pi);
                    is_policy_changed = policy_improvement(A, V, pi);
                    if (is_policy_changed == 0) {
                        break;
                    }
                    k = k + 1;
                }
                t = math21_time_getticks() - t;
                std::cout << "Optimal policy is reached after "
                          << k << " iterations in " << t << " seconds\n";

                V.getTensor().log("optimal_value");
                f.draw(V.getTensor(), "optimal_value");

            }

            void test(ImageDraw &f) {
                init();
                policy_iteration(f);
            }
        }
    }

    void math21_rl_chapter04_car_rental_synchronous(ImageDraw &f) {
        m21log("car rental synchronous");
        rl::car_rental_synchronous::test(f);
    }
}