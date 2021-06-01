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
#include "random_walk.h"

namespace math21 {

    namespace rl {
        namespace random_walk {
            IndexFunctional<NumR> v_pi;
            IndexFunctional<NumR> V0;
            NumN al;
            NumN ar;
            RanBinomial ran;
            NumN S_size;
            NumN S_plus_size;
            const NumN method_TD = 1;
            const NumN method_MC = 2;

            void init() {
                S_size = 5;
                S_plus_size = 7;

                v_pi.setSize(S_plus_size);
                v_pi.setStartIndex(0);
                math21_operator_container_set_value(v_pi.getTensor(), 0.0, 1.0 / (S_plus_size - 1));
                v_pi(S_plus_size - 1) = 0;

                V0.setSize(S_plus_size);
                V0.setStartIndex(0);
                V0.getTensor() = 0.5;
                V0(0) = 0;
                V0(S_plus_size - 1) = 0;

                al = 0;
                ar = 1;

                ran.set(1, 0.5);
            }

            void TD_0(IndexFunctional<NumR> &V,
                      Seqce<NumN> &traj, Seqce<NumR> &rs,
                      NumR alpha = 0.1, NumB isBatch = 0) {
                NumN s = 3;
                NumN s_prime;

                traj.clear();
                rs.clear();

                traj.push(s);
                NumR r;
                while (1) {
                    NumN x;
                    math21_random_draw(x, ran);
                    if (x == al) {
                        s_prime = s - 1;
                    } else {
                        s_prime = s + 1;
                    }
                    if (s_prime == S_plus_size - 1) {
                        r = 1;
                    } else {
                        r = 0;
                    }
                    if (!isBatch) {
                        V(s) = V(s) + alpha * (r + V(s_prime) - V(s));
                    }
                    s = s_prime;
                    traj.push(s);
                    rs.push(r);
                    if (s == 0 || s == S_plus_size - 1) {
                        break;
                    }
                }
            }

            void constant_alpha_MC(IndexFunctional<NumR> &V,
                                   Seqce<NumN> &traj, Seqce<NumR> &rs,
                                   NumR alpha = 0.1, NumB isBatch = 0) {
                NumN s = 3;
                NumN s_prime;

                traj.clear();
                rs.clear();

                traj.push(s);
                NumR G;
                while (1) {
                    NumN x;
                    math21_random_draw(x, ran);
                    if (x == al) {
                        s_prime = s - 1;
                    } else {
                        s_prime = s + 1;
                    }
                    if (s_prime == S_plus_size - 1) {
                        G = 1;
                    } else {
                        G = 0;
                    }
                    s = s_prime;
                    traj.push(s);
                    if (s == 0 || s == S_plus_size - 1) {
                        break;
                    }
                }
                if (!isBatch) {
                    for (NumN i = 1; i <= traj.size() - 1; ++i) {
                        s = traj(i);
                        V(s) = V(s) + alpha * (G - V(s));
                    }
                }
                rs.setSize(traj.size() - 1);
                rs = G;
            }

            // example 6.2 left
            void compute_state_value(ImageDraw &f) {
                VecN episodes_vec(4);
                episodes_vec = 0, 1, 10, 100;
                SetN episodes;
                episodes.add(episodes_vec);

                IndexFunctional<NumR> V;
                V = V0;
                f.setDir("figure_6_2_left");
                f.reset();
                Seqce<NumN> traj;
                Seqce<NumR> rs;
                f.plot(v_pi.getTensor(), "true values", 0);
                f.save("true values");
                for (NumN i = 0; i <= 100; ++i) {
                    if (episodes.contains(i)) {
                        std::string name = math21_string_to_string(i) + " episodes";
                        f.plot(V.getTensor(), name.c_str(), 0);
                        f.save(name.c_str());
                    }
                    TD_0(V, traj, rs);
                }
                f.save("estimated_value");
            }

            // example 6.2 right
            void rms_error(ImageDraw &f) {
                VecR TD_0_alphaes(3);
                TD_0_alphaes = 0.15, 0.1, 0.05;
                VecR MC_alphaes(4);
                MC_alphaes = 0.01, 0.02, 0.03, 0.04;
                NumN episodes = 101;
                NumN runs = 100;
                VecR total_errors(episodes);
                total_errors = 0;
                VecR errors(episodes);
                NumN method;
                NumR alpha;
                IndexFunctional<NumR> V;

                Seqce<NumN> traj;
                Seqce<NumR> rs;
                f.setDir("figure_6_2_right");
                f.reset();
                for (NumN i = 1; i <= TD_0_alphaes.size() + MC_alphaes.size(); ++i) {
                    if (i <= TD_0_alphaes.size()) {
                        method = method_TD;
                        alpha = TD_0_alphaes(i);
                    } else {
                        method = method_MC;
                        alpha = MC_alphaes(i - TD_0_alphaes.size());
                    }
                    for (NumN r = 1; r <= runs; ++r) {
                        V = V0;
                        for (NumN j = 1; j <= episodes; ++j) {
                            NumR err;
                            err = math21_operator_container_distance(V.getTensor(), v_pi.getTensor(), 2);
                            err = err / xjsqrt(S_size);
                            errors(j) = err;
                            if (method == method_TD) {
                                TD_0(V, traj, rs, alpha);
                            } else {
                                constant_alpha_MC(V, traj, rs, alpha);
                            }
                        }
                        math21_operator_container_addToA(total_errors, errors);
                    }
                    math21_operator_container_linear_to_A(1.0 / runs, total_errors);
                    std::string name;
                    if (method == method_TD) {
                        name = "TD_0 " + math21_string_to_string(alpha);
                    } else {
                        name = "constant_alpha_MC " + math21_string_to_string(alpha);
                    }
                    f.plot(total_errors, name.c_str(), 0);
                }
                f.save("RMS");
            }

            void batch_updating(NumN method, VecR &total_errors, NumN episodes, NumR alpha = 1e-3) {
                NumN runs = 100;
                if (total_errors.isSameSize(episodes) == 0) {
                    total_errors.setSize(episodes);
                }
                total_errors = 0;
                VecR errors(episodes);
                IndexFunctional<NumR> V;
                Seqce<Seqce<NumN>> trajs;
                Seqce<Seqce<NumR>> rss;
                Seqce<NumN> traj;
                Seqce<NumR> rs;

                IndexFunctional<NumR> sum_delta;
                sum_delta.setSize(S_plus_size);
                sum_delta.setStartIndex(0);

                for (NumN r = 1; r <= runs; ++r) {
                    V = V0;
                    trajs.clear();
                    rss.clear();
                    for (NumN ep = 1; ep <= episodes; ++ep) {
                        if (method == method_TD) {
                            TD_0(V, traj, rs, 0.1, 1);
                        } else {
                            constant_alpha_MC(V, traj, rs, 0.1, 1);
                        }
                        trajs.push(traj);
                        rss.push(rs);
                        while (1) {
                            sum_delta.getTensor() = 0;
                            for (NumN j = 1; j <= trajs.size(); ++j) {
                                const Seqce<NumN> &traj = trajs(j);
                                const Seqce<NumR> rs = rss(j);
                                for (NumN i = 1; i <= traj.size() - 1; ++i) {
                                    NumN s, s_prime;
                                    s = traj(i);
                                    s_prime = traj(i + 1);
                                    NumR delta;
                                    if (method == method_TD) {
                                        NumR r = rs(i);
                                        delta = r + V(s_prime) - V(s);
                                    } else {
                                        NumR G;
                                        G = rs(i);
                                        delta = G - V(s);
                                    }
                                    sum_delta(s) = sum_delta(s) + delta;
                                }
                            }

                            if (math21_operator_container_norm(sum_delta.getTensor(), 1) < 1) {
                                break;
                            }
                            math21_operator_container_linear_to_A(1, V.getTensor(), alpha, sum_delta.getTensor());
                        }
                        NumR err;
                        err = math21_operator_container_distance(V.getTensor(), v_pi.getTensor(), 2);
                        err = err / xjsqrt(S_size);
                        errors(ep) = err;
                    }
                    math21_operator_container_addToA(total_errors, errors);
                }
                math21_operator_container_linear_to_A(1.0 / runs, total_errors);
            }

            void example_6_2(ImageDraw &f) {
                compute_state_value(f);
                rms_error(f);
            }

            void figure_6_3(ImageDraw &f) {
                NumN episodes = 100;
                VecR TD_errors;
                VecR MC_errors;
                batch_updating(method_TD, TD_errors, episodes);
                batch_updating(method_MC, MC_errors, episodes);
                f.setDir("figure_6_3");
                f.reset();
                f.plot(TD_errors, "TD_errors", 0);
                f.plot(MC_errors, "MC_errors", 0);
                f.save("batch_updating");
            }

            void test(ImageDraw &f) {
                init();
                example_6_2(f);
                figure_6_3(f);
            }
        }
    }

    void math21_rl_chapter06_random_walk(ImageDraw &f) {
        m21log("random_walk");
        rl::random_walk::test(f);
    }
}