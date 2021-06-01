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
#include "cliff_walking.h"

namespace math21 {

    namespace rl {
        namespace cliff_walking {
            const NumN X_MAX = 12;
            const NumN Y_MAX = 4;
            const NumR para_epsilon = 0.1;
            const NumR para_alpha = 0.5;
            const NumR para_gamma = 1;
            const NumN al = 1;
            const NumN ar = 2;
            const NumN au = 3;
            const NumN ad = 4;
            VecN START;
            VecN GOAL;
            DefaultRandomEngine engine(21);
            RanBinomial binomial;
            RanUniform uniform;
            NumN A_size;

            void init() {
                START.setSize(2);
                START = 1, 1;
                GOAL.setSize(2);
                GOAL = X_MAX, 1;

                A_size = 4;

                binomial.set(1, para_epsilon);
                uniform.set(1, A_size);
            }

            NumR step(const VecN &s, NumN a, VecN &s_prime) {
                MATH21_ASSERT(!math21_operator_container_isEqual(s, GOAL))
                NumN x, y;
                x = s(1);
                y = s(2);
                if (a == al) {
                    x = x - 1;
                } else if (a == ar) {
                    x = x + 1;
                } else if (a == au) {
                    y = y + 1;
                } else {
                    y = y - 1;
                }
                if (x < 1) {
                    x = 1;
                } else if (x > X_MAX) {
                    x = X_MAX;
                }
                if (y < 1) {
                    y = 1;
                } else if (y > Y_MAX) {
                    y = Y_MAX;
                }
                s_prime = x, y;
                NumR r;
                r = -1;
                if (xjIsIn(x, 2, X_MAX - 1) && xjIsIn(y, 1, 1)) {
                    s_prime = START;
                    r = -100;
                }
                return r;
            }

            void print_optimal_policy(const TenR &Q, const char *name = 0) {
                VecN index(Q.dims());
                index = 0, 0, 1;
                TensorFunction_argmax f_arg_max;
                MatN pi;
                math21_operator_tensor_f_shrink(Q, pi, index, f_arg_max);

                Tensor<char> pi_string;
                pi_string.setSize(pi.shape());

                VecN s(2);
                for (NumN i = 1; i <= pi.dim(1); ++i) {
                    for (NumN j = 1; j <= pi.dim(2); ++j) {
                        s = i, j;
                        if (math21_operator_container_isEqual(s, GOAL)) {
                            pi_string(s) = 'G';
                            continue;
                        }
                        if (xjIsIn(i, 2, X_MAX - 1) && xjIsIn(j, 1, 1)) {
                            pi_string(s) = '.';
                            continue;
                        }
                        NumN a = pi(s);
                        if (a == al) {
                            pi_string(s) = '<';
                        } else if (a == ar) {
                            pi_string(s) = '>';
                        } else if (a == au) {
                            pi_string(s) = '^';
                        } else {
                            pi_string(s) = 'V';
                        }
                    }
                }

                if (name == 0) {
                    name = "pi";
                }
                Tensor<char> pi_draw;
                math21_operator_matrix_axis_to_image(pi_string, pi_draw);
                pi_draw.log(name);
            }

            NumN choose_action(const VecN &s, const TenR &Q) {
                NumN sample;
                binomial.draw(sample);
                NumN a;
                if (sample == 1) {
                    NumN ia;
                    uniform.draw(ia);
                    a = ia;
                } else {
                    VecN X(3);
                    X = s(1), s(2), 0;
                    TensorView<NumR> Q_at_s = Q.shrinkView(X);
                    a = math21_operator_container_argmax_random(Q_at_s, engine);
                }
                return a;
            }

            NumR Sarsa(TenR &Q, NumB is_expected = 0, NumR alpha = para_alpha) {
                VecN s;
                s.copyFrom(START);
                NumN a;
                a = choose_action(s, Q);
                NumR sum_r;
                sum_r = 0;
                VecN s_prime(2);
                while (1) {
                    NumR r = step(s, a, s_prime);
                    NumN a_prime = choose_action(s_prime, Q);
                    sum_r = sum_r + r;
                    NumR G;
                    if (!is_expected) {
                        G = Q(s_prime, a_prime);
                    } else {
                        G = 0;
                        VecN X(3);
                        X = s_prime(1), s_prime(2), 0;
                        TensorView<NumR> Q_at_s_prime = Q.shrinkView(X);
                        NumR max = math21_operator_container_max(Q_at_s_prime);
                        Set set_a;
                        math21_operator_container_argwhere(Q_at_s_prime, max, set_a);
                        for (NumN ia = 1; ia <= A_size; ++ia) {
                            NumN a = ia;
                            NumR p;
                            if (set_a.contains(a)) {
                                p = (1 - para_epsilon) / set_a.size() + para_epsilon / A_size;
                            } else {
                                p = para_epsilon / A_size;
                            }
                            G = G + p * Q(s_prime, a);
                        }
                    }
                    G = r + para_gamma * G;
                    Q(s, a) = Q(s, a) + alpha * (G - Q(s, a));
                    s = s_prime;
                    a = a_prime;
                    if (math21_operator_container_isEqual(s, GOAL)) {
                        break;
                    }
                }
                return sum_r;
            }

            NumR Q_learning(TenR &Q, NumR alpha = para_alpha) {
                VecN s;
                s.copyFrom(START);
                NumR sum_r;
                sum_r = 0;
                VecN s_prime(2);
                while (1) {
                    NumN a;
                    a = choose_action(s, Q);
                    NumR r;
                    r = step(s, a, s_prime);
                    sum_r = sum_r + r;

                    VecN X(3);
                    X = s_prime(1), s_prime(2), 0;
                    TensorView<NumR> Q_at_s_prime = Q.shrinkView(X);
                    NumR max = math21_operator_container_max(Q_at_s_prime);

                    Q(s, a) = Q(s, a) + alpha * (r + para_gamma * max - Q(s, a));
                    s = s_prime;
                    if (math21_operator_container_isEqual(s, GOAL)) {
                        break;
                    }
                }
                return sum_r;
            }

            void figure_6_4(ImageDraw &f) {
                NumN episodes = 500;
                NumN runs = 50;
                VecR sum_rs_Sarsa(episodes);
                VecR sum_rs_Q_learning(episodes);
                sum_rs_Sarsa = 0;
                sum_rs_Q_learning = 0;
                TenR Q_Sarsa(X_MAX, Y_MAX, A_size);
                TenR Q_Q_learning(X_MAX, Y_MAX, A_size);
                for (NumN run = 1; run <= runs; ++run) {
                    Q_Sarsa = 0;
                    Q_Q_learning = 0;
                    for (NumN i = 1; i <= episodes; ++i) {
                        sum_rs_Sarsa(i) += Sarsa(Q_Sarsa);
                        sum_rs_Q_learning(i) += Q_learning(Q_Q_learning);
                    }
                }
                math21_operator_container_linear_to_A(1.0 / runs, sum_rs_Sarsa);
                math21_operator_container_linear_to_A(1.0 / runs, sum_rs_Q_learning);

                f.setDir("figure_6_4");
                f.reset();
                f.plot(sum_rs_Sarsa, "Sarsa", 0);
                f.plot(sum_rs_Q_learning, "Q-learning", 0);
                f.save("sum_r");

                print_optimal_policy(Q_Sarsa, "Sarsa");
                print_optimal_policy(Q_Q_learning, "Q-learning");
            }

            void figure_6_6(ImageDraw &f) {
                NumN episodes = 1000;
                NumN runs = 10;
                VecR alphas(10);
                math21_operator_container_set_value(alphas, 0.1, 0.1);
                NumN asymptotic_Sarsa = 1;
                NumN asymptotic_expected_Sarsa = 2;
                NumN asymptotic_Q_learning = 3;
                NumN interim_Sarsa = 4;
                NumN interim_expected_Sarsa = 5;
                NumN interim_Q_learning = 6;
                NumN methods_size = 6;
                MatR performance(methods_size, alphas.size());
                performance = 0;
                for (NumN run = 1; run <= runs; ++run) {
//                    m21log("run", run);
                    for (NumN i = 1; i <= alphas.size(); ++i) {
                        NumR alpha = alphas(i);
                        TenR Q_Sarsa(X_MAX, Y_MAX, A_size);
                        TenR Q_expected_Sarsa(X_MAX, Y_MAX, A_size);
                        TenR Q_Q_learning(X_MAX, Y_MAX, A_size);
                        Q_Sarsa = 0;
                        Q_expected_Sarsa = 0;
                        Q_Q_learning = 0;
                        for (NumN ep = 1; ep <= episodes; ++ep) {
                            NumR sum_rs_Sarsa;
                            NumR sum_rs_expected_Sarsa;
                            NumR sum_rs_Q_learning;
                            sum_rs_Sarsa = Sarsa(Q_Sarsa, 0, alpha);
                            sum_rs_expected_Sarsa = Sarsa(Q_Sarsa, 1, alpha);
                            sum_rs_Q_learning = Q_learning(Q_Q_learning, alpha);
                            performance(asymptotic_Sarsa, i) += sum_rs_Sarsa;
                            performance(asymptotic_expected_Sarsa, i) += sum_rs_expected_Sarsa;
                            performance(asymptotic_Q_learning, i) += sum_rs_Q_learning;
                            if (ep <= 100) {
                                performance(interim_Sarsa, i) += sum_rs_Sarsa;
                                performance(interim_expected_Sarsa, i) += sum_rs_expected_Sarsa;
                                performance(interim_Q_learning, i) += sum_rs_Q_learning;
                            }
                        }
                    }
                }

                Seqce<VecN> X;
                X.setSize(performance.dims());
                X(1).setSize(3);
                X(2).setSize(1);
                X(1) = 1, 2, 3;
                X(2) = 0;

                TensorSub<NumR> sub_performance = performance.sliceSub(X);
                math21_operator_container_linear_to_A(1.0 / (runs * episodes), sub_performance);
                X(1) = 4, 5, 6;
                X(2) = 0;
                TensorSub<NumR> sub_performance2 = performance.sliceSub(X);
                math21_operator_container_linear_to_A(1.0 / (runs * 100), sub_performance2);

                performance.log("performance");
                f.setDir("figure_6_6");
                f.reset();
                const char *name = "sum_reward_per_episode";
                f.plot(performance, name);
                f.save(name);
            }

            void test(ImageDraw &f) {
                init();
//                figure_6_4(f);
                figure_6_6(f);
            }
        }
    }

    void math21_rl_chapter06_cliff_walking(ImageDraw &f) {
        m21log("cliff_walking");
        rl::cliff_walking::test(f);
    }
}