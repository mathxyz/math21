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
#include "gamblers_problem.h"

namespace math21 {
    namespace rl {
        namespace gamblers_problem {
            DefaultRandomEngine engine(21);
            NumN Goal = 100;
            VecN S;
            VecN S_plus;
            NumR ph = 0.4;

            void init() {
                S.setSize(Goal + 1);
                math21_operator_container_letters(S, 0);
                S_plus.setSize(Goal - 1);
                math21_operator_container_letters(S_plus, 1);
            }

            void figure_4_3(ImageDraw &f) {
                Dict<NumN, NumR> V;
                V.add(S, 0);
                V.valueAt(Goal) = 1;
                Seqce<Dict<NumN, NumR>> sweeps_history;
                NumR delta;
                VecN A;
                VecR vs;

                // value iteration
                while (1) {
                    sweeps_history.push(V);
                    delta = 0;
                    for (NumN i = 1; i <= S_plus.size(); ++i) {
                        NumN s = S_plus(i);
                        NumR v = V.valueAt(s);
                        A.setSize(xjmin(s, Goal - s) + 1);
                        math21_operator_container_letters(A, 0);
                        vs.setSize(A.size());
                        for (NumN ia = 1; ia <= A.size(); ++ia) {
                            NumN a = A(ia);
                            vs(ia) = ph * V.valueAt(s + a) + (1 - ph) * V.valueAt(s - a);
                        }
                        V.valueAt(s) = math21_operator_container_max(vs);
                        delta = xjmax(delta, xjabs(v - V.valueAt(s)));
                    }
                    if (delta < 1e-9) {
                        sweeps_history.push(V);
                        break;
                    }
                }

                // compute optimal policy
                Dict<NumN, NumN> pi;
                pi.add(S, 0);
                for (NumN i = 1; i <= S_plus.size(); ++i) {
                    NumN s = S_plus(i);
                    A.setSize(xjmin(s, Goal - s) + 1);
                    math21_operator_container_letters(A, 0);
                    vs.setSize(A.size());
                    for (NumN ia = 1; ia <= A.size(); ++ia) {
                        NumN a = A(ia);
                        vs(ia) = ph * V.valueAt(s + a) + (1 - ph) * V.valueAt(s - a);
                    }
                    // must choose a > 0
                    // round first, so we can choose the smallest a when there are several approximately largest values.
                    math21_operator_container_round_to(vs, 6);
                    pi.valueAt(s) = A(math21_operator_container_argmax(vs, 2));
                }

                // figure
                TenR V_draw;
                TenR pi_draw;
                math21_operator_container_map_1d_to_tensor(sweeps_history, V_draw);
                math21_operator_map_1d_to_tensor(pi, pi_draw);

//                V_draw.log("value_estimations");
//                pi_draw.log("final_policy");

                f.setDir("figure_4_3");
                f.draw(V_draw, "value_estimations");
                f.draw(pi_draw, "final_policy", 0);

            }

            void test(ImageDraw &f) {
                init();
                figure_4_3(f);
            }
        }
    }

    void math21_rl_chapter04_gamblers_problem(ImageDraw &f) {
        m21log("gambler's problem");
        rl::gamblers_problem::test(f);
    }
}