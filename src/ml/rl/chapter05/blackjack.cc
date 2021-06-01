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
#include "blackjack.h"

namespace math21 {

    namespace rl {
        namespace blackjack {
            NumN ah = 1; // hit
            NumN as = 2; // stick

            VecN PI_P;
            VecN PI_D;
            DefaultRandomEngine engine(21);
            RanUniform ranUniform_card;
            RanUniform ranUniform_initial;
            // s = ( ua_p, sum_p, c1_d)

            struct state_action_pair {
            private:
                void set(const VecN &s0, NumN a0) {
                    s.copyFrom(s0);
                    a = a0;
                }

            public:
                VecN s;
                NumN a;

                state_action_pair(const VecN &s, NumN a) {
                    set(s, a);
                }

                state_action_pair(const state_action_pair &pair) {
                    *this = pair;
                }

                state_action_pair &operator=(const state_action_pair &pair) {
                    set(pair.s, pair.a);
                    return *this;
                }

                virtual ~state_action_pair() {
                }
            };

            struct policy {
            public:
                policy() {
                }

                virtual ~policy() {
                }

                virtual NumN getAction(const VecN &s) const = 0;

                virtual NumR getProbability(const VecN &s, NumN a) const {
                    return 1;
                }

                virtual void setAction(const VecN &s, NumN a) {

                }
            };

            struct policy_pi : public policy {
            private:
                IndexFunctional<NumN> pi;

                void init() {
                    pi.setSize(2, 10, 10);
                    pi.setStartIndex(0, 12, 1);
                }

            public:
                policy_pi() {
                    init();
                }

                const IndexFunctional<NumN> &get() const {
                    return pi;
                }

                IndexFunctional<NumN> &get() {
                    return pi;
                }

                policy_pi(const policy &other) {
                    init();
                    *this = other;
                }

                policy &operator=(const policy &other) {
                    VecN start(3);
                    pi.get_start_index(start);
                    VecN end(3);
                    pi.get_end_index(end);
                    VecN s(3);
                    s.assign(start);
                    while (1) {
                        pi(s) = other.getAction(s);
                        if (math21_operator_container_increaseNumFromRight(end, s, start) == 0) {
                            break;
                        }
                    }
                    return *this;
                }

                virtual ~policy_pi() {
                }

                NumN getAction(const VecN &s) const override {
                    return pi.operator()(s);
                }

                NumR getProbability(const VecN &s, NumN a) const override {
                    return 1;
                }

                void setAction(const VecN &s, NumN a) override {
                    pi(s) = a;
                }
            };

            struct policy_pi_p : public policy {
            public:
                policy_pi_p() {
                }

                virtual ~policy_pi_p() {
                }

                NumN getAction(const VecN &s) const override {
                    NumN sum_p = s(2);
                    return PI_P.operator()(sum_p);
                }

                NumR getProbability(const VecN &s, NumN a) const override {
                    return 1;
                }
            };

            struct policy_mu_p : public policy {
            private:
                NumR p = 0.5;
            public:
                policy_mu_p() {
                }

                virtual ~policy_mu_p() {
                }

                NumN getAction(const VecN &s) const override {
                    RanBinomial ran(1, p);
                    NumN k;
                    math21_random_draw(k, ran);
                    if (k == 1) {
                        return as;
                    } else {
                        return ah;
                    }
                }

                NumR getProbability(const VecN &s, NumN a) const override {
                    return p;
                }
            };

            void init() {
                PI_P.setSize(21);
                math21_operator_container_set_num(PI_P, ah, 1, 11);
                math21_operator_container_set_num(PI_P, ah, 12, 19);
                math21_operator_container_set_num(PI_P, as, 20, 21);

                PI_D.setSize(21);
                math21_operator_container_set_num(PI_D, ah, 1, 11);
                math21_operator_container_set_num(PI_D, ah, 12, 16);
                math21_operator_container_set_num(PI_D, as, 17, 21);

                ranUniform_card.set(1, 13);
            }

            NumN get_card() {
                NumN x;
                math21_random_draw(x, ranUniform_card);
                return xjmin(x, 10);
            }

            NumN card_value(NumN card) {
                if (card == 1) {
                    return 11;
                } else {
                    return card;
                }
            }

            // s = ( ua_p, sum_p, c1_d)
            NumR play(const policy &pi, Seqce<state_action_pair> &traj, const VecN &s0, const NumN a0,
                      NumB is_using_s0 = 0, NumB is_using_a0 = 0) {
                NumN sum_p = 0;
                NumN sum_d = 0;
                NumB ua_p = 0;
                NumN c1_d = 0;
                NumN c2_d = 0;
                NumB ua_d = 0;
                NumN card;
                VecN s(3);
                NumN a;
                NumR R;
                traj.clear();
                if (!is_using_s0) {
                    while (sum_p < 12) {
                        card = get_card();
                        sum_p = sum_p + card_value(card);
                        if (card == 1) {
                            if (sum_p > 21) {
                                sum_p = sum_p - 10;
                            } else {
                                ua_p = 1;
                            }
                        }
                    }
                    c1_d = get_card();
                    c2_d = get_card();
                } else {
                    ua_p = s0(1);
                    sum_p = s0(2);
                    c1_d = s0(3);
                    c2_d = get_card();
                    MATH21_ASSERT(sum_p >= 12 && sum_p <= 21)
                }
                sum_d = card_value(c1_d) + card_value(c2_d);
                if (c1_d == 1 || c2_d == 1) {
                    ua_d = 1;
                } else {
                    ua_d = 0;
                }

                if (sum_d > 21) {
                    MATH21_ASSERT(sum_d == 22)
                    sum_d = sum_d - 10;
                }

                // game starts!
                // player's turn.
                while (1) {
                    s = ua_p, sum_p, c1_d;
                    if (is_using_a0) {
                        a = a0;
                        is_using_a0 = 0;
                    } else {
                        a = pi.getAction(s);
                    }
                    traj.push(state_action_pair(s, a));

                    if (a == as) {
                        break;
                    }
                    card = get_card();
                    sum_p = sum_p + card_value(card);
                    if (card == 1) {
                        if (ua_p == 1) {
                            sum_p = sum_p - 10;
                        } else {
                            ua_p = 1;
                        }
                    }
                    if (sum_p > 21) {
                        if (ua_p == 1) {
                            sum_p = sum_p - 10;
                            ua_p = 0;
                        }
                    }
                    if (sum_p > 21) {
                        return -1;
                    }
                }

                // dealer's turn
                while (1) {
                    a = PI_D(sum_d);
                    if (a == as) {
                        break;
                    }
                    card = get_card();
                    sum_d = sum_d + card_value(card);
                    if (card == 1) {
                        if (ua_d == 1) {
                            sum_d = sum_d - 10;
                        } else {
                            ua_d = 1;
                        }
                    }
                    if (sum_d > 21) {
                        if (ua_d == 1) {
                            sum_d = sum_d - 10;
                            ua_d = 0;
                        }
                    }
                    if (sum_d > 21) {
                        return 1;
                    }
                }

                // compare sum_p and sum_d
                if (sum_p > sum_d) {
                    R = 1;
                } else if (sum_p == sum_d) {
                    R = 0;
                } else {
                    R = -1;
                }
                return R;
            }


            // s = (ua_p, sum_p, c1_d)
            void MC_policy_evaluation_on_policy(IndexFunctional<NumR> &V, NumN episodes) {
                if (V.isEmpty()) {
                    V.setSize(2, 10, 10);
                    V.setStartIndex(0, 12, 1);
                    V.getTensor() = 0;
                } else {
                    MATH21_ASSERT(V.getTensor().isSameSize(2, 10, 10))
                    V.getTensor() = 0;
                }
                IndexFunctional<NumN> n;
                n.setSize(2, 10, 10);
                n.setStartIndex(0, 12, 1);
                n.getTensor() = 0;
                NumN i, j;
                policy_pi_p pi_p;
                Seqce<state_action_pair> traj;
                VecN s0(3);
                NumN a0 = 0;
                for (i = 1; i <= episodes; ++i) {
                    NumR G;
                    G = play(pi_p, traj, s0, a0, 0, 0);
                    for (j = 1; j <= traj.size(); ++j) {
                        const state_action_pair &pair = traj(j);
                        const VecN &s = pair.s;
                        const NumN &a = pair.a;
                        n(s) = n(s) + 1;
                        V(s) = V(s) + (G - V(s)) / n(s);
                    }
                }
            }

            template<typename T>
            void math21_tool_tensor_log_3d_as_2d(const Tensor<T> &A, const char *name) {
                MATH21_ASSERT(A.dims() == 3)
                VecN X(3);
                if (name == 0) {
                    name = "";
                }
                for (NumN i = 1; i <= A.dim(1); ++i) {
                    X = i, 0, 0;
                    TensorView<T> B = A.shrinkView(X);
                    const char *name2 = (math21_string_to_string(name) + "_" + math21_string_to_string(i)).c_str();
                    B.log(name2);
                }
            }

            // MC with exploring starts
            void MC_es(IndexFunctional<NumR> &Q, policy_pi &pi, NumN episodes) {
                if (Q.isEmpty()) {
                    Q.setSize(2, 10, 10, 2);
                    Q.setStartIndex(0, 12, 1, ah);
                    Q.getTensor() = 0;
                } else {
                    MATH21_ASSERT(Q.getTensor().isSameSize(2, 10, 10, 2))
                    Q.getTensor() = 0;
                }
                IndexFunctional<NumN> n;
                n.setSize(2, 10, 10, 2);
                n.setStartIndex(0, 12, 1, ah);
                n.getTensor() = 0;
                policy_pi_p pi_p;
                pi = pi_p;

                NumN i, j;
                VecN s0(3);
                NumN a0 = 0;
                ranUniform_initial.set(1, Q.size());
                VecN index(4);
                NumR G;
                Seqce<state_action_pair> traj;
//                math21_tool_tensor_log_3d_as_2d(pi.get().getTensor(), "pi");
                for (i = 1; i <= episodes; ++i) {
                    NumN k;
                    math21_random_draw(k, ranUniform_initial);
                    Q.getIndex(k, index);
                    s0 = index(1), index(2), index(3);
                    a0 = index(4);
                    G = play(pi, traj, s0, a0, 1, 1);
                    for (j = 1; j <= traj.size(); ++j) {
                        const state_action_pair &pair = traj(j);
                        const VecN &s = pair.s;
                        const NumN &a = pair.a;
                        n(s, a) = n(s, a) + 1;
                        Q(s, a) = Q(s, a) + (G - Q(s, a)) / (n(s, a));

                        // pi(s) = argmax(Q(s,a)) for a.
                        const TenR &tenQ = Q.getTensor();
                        VecN X;
                        X.setSize(tenQ.dims());
                        X =
                                s(1) + Q.getTensor_index_offset(1),
                                s(2) + Q.getTensor_index_offset(2),
                                s(3) + Q.getTensor_index_offset(3),
                                0;
                        TenR Q_at_s;
                        math21_operator_tensor_shrink(tenQ, Q_at_s, X);
                        NumZ a_offset = -Q.getTensor_index_offset(4);
//                        Q_at_s.log("Q_at_s");
                        pi.setAction(s, a_offset + math21_operator_container_argmax_random(Q_at_s, engine));
                    }
                }
            }

            // MC with exploring starts
            void MC_es_without_incremental(IndexFunctional<NumR> &Q, policy_pi &pi, NumN episodes) {
                if (Q.isEmpty()) {
                    Q.setSize(2, 10, 10, 2);
                    Q.setStartIndex(0, 12, 1, ah);
                    Q.getTensor() = 0;
                } else {
                    MATH21_ASSERT(Q.getTensor().isSameSize(2, 10, 10, 2))
                    Q.getTensor() = 0;
                }
                IndexFunctional<NumN> n;
                n.setSize(2, 10, 10, 2);
                n.setStartIndex(0, 12, 1, ah);
                n.getTensor() = 1;
                policy_pi_p pi_p;
                pi = pi_p;

                NumN i, j;
                VecN s0(3);
                NumN a0 = 0;
                ranUniform_initial.set(1, Q.size());
                VecN index(4);
                NumR G;
                Seqce<state_action_pair> traj;
//                math21_tool_tensor_log_3d_as_2d(pi.get().getTensor(), "pi");
                for (i = 1; i <= episodes; ++i) {
                    NumN k;
                    math21_random_draw(k, ranUniform_initial);
                    Q.getIndex(k, index);
                    s0 = index(1), index(2), index(3);
                    a0 = index(4);
                    G = play(pi, traj, s0, a0, 1, 1);
                    for (j = 1; j <= traj.size(); ++j) {
                        const state_action_pair &pair = traj(j);
                        const VecN &s = pair.s;
                        const NumN &a = pair.a;
                        n(s, a) = n(s, a) + 1;
                        Q(s, a) = Q(s, a) + G;

                        // pi(s) = argmax(Q(s,a)) for a.
                        const TenR &tenQ = Q.getTensor();
                        VecN X;
                        X.setSize(tenQ.dims());
                        X =
                                s(1) + Q.getTensor_index_offset(1),
                                s(2) + Q.getTensor_index_offset(2),
                                s(3) + Q.getTensor_index_offset(3),
                                0;
                        TenR Q_at_s;
                        TenN n_at_s;
                        math21_operator_tensor_shrink(tenQ, Q_at_s, X);
                        math21_operator_tensor_shrink(n.getTensor(), n_at_s, X);
                        NumZ a_offset = -Q.getTensor_index_offset(4);
//                        Q_at_s.log("Q_at_s");
                        math21_operator_container_divide_to_A(Q_at_s, n_at_s);
                        pi.setAction(s, a_offset + math21_operator_container_argmax_random(Q_at_s, engine));
                    }
                }
                math21_operator_container_divide_to_A(Q.getTensor(), n.getTensor());
            }

            void MC_estimate_state_value_off_policy(VecR &OS, VecR &WS, NumN episodes) {
                NumN i, j;
                VecN s0(3);
                NumN a0 = 0;
                s0 = 1, 13, 2;
                NumN n = 0;
                NumR Vo = 0;
                NumR Vw = 0;
                NumR W = 0;
                NumR C = 0;
                MATH21_ASSERT(OS.size() == episodes)
                MATH21_ASSERT(WS.size() == episodes)
                Seqce<state_action_pair> traj;
                NumR G;
                policy_pi_p pi_p;
                policy_mu_p mu_p;
                for (i = 1; i <= episodes; ++i) {
                    G = play(mu_p, traj, s0, a0, 1, 0);
                    W = 1;
                    for (j = 1; j <= traj.size(); ++j) {
                        const state_action_pair &pair = traj(j);
                        const VecN &s = pair.s;
                        const NumN &a = pair.a;
                        if (a == pi_p.getAction(s)) {
                            W = W * (1 / mu_p.getProbability(s, a));
                        } else {
                            W = 0;
                            break;
                        }
                    }
                    n = n + 1;
                    C = C + W;
                    Vo = Vo + (W * G - Vo) / n;
                    if (C != 0) {
                        Vw = Vw + W * (G - Vw) / C;
                    }
                    OS(i) = Vo;
                    WS(i) = Vw;
                }
            }

            void figure_5_1() {
                IndexFunctional<NumR> V1, V2;
                MC_policy_evaluation_on_policy(V1, (NumN) xjpow(10, 4));
                MC_policy_evaluation_on_policy(V2, 5 * (NumN) xjpow(10, 5));

                // log
                MatR V_log;
                VecN X;
                X.setSize(3);
                X = 2, 0, 0;
                math21_operator_tensor_shrink(V1.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("Usable Ace, 10000 Episodes");
                math21_operator_tensor_shrink(V2.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("Usable Ace, 500000 Episodes");
                X = 1, 0, 0;
                math21_operator_tensor_shrink(V1.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("No Usable Ace, 10000 Episodes");
                math21_operator_tensor_shrink(V2.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("No Usable Ace, 500000 Episodes");
            }

            void figure_5_2() {
                IndexFunctional<NumR> Q;
                policy_pi pi;
                MC_es(Q, pi, 5 * (NumN) xjpow(10, 5));
//                MC_es_without_incremental(Q, pi, 5 * (NumN) xjpow(10, 4));

                IndexFunctional<NumR> V;
                V.setSize(2, 10, 10);
                V.setStartIndex(0, 12, 1);

                VecN X;
                X.setSize(4);
                X = 0, 0, 0, 1;
                TensorFunction_max f_max;
                math21_operator_tensor_f_shrink(Q.getTensor(), V.getTensor(), X, f_max);

                // log
                MatR V_log;
                MatN pi_log;
                X.setSize(3);
                X = 2, 0, 0;
                math21_operator_tensor_shrink(pi.get().getTensor(), pi_log, X);
                math21_operator_matrix_reverse_x_axis(pi_log);
                pi_log.log("Optimal policy with usable Ace");
                math21_operator_tensor_shrink(V.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("Optimal value with usable Ace");

                X = 1, 0, 0;
                math21_operator_tensor_shrink(pi.get().getTensor(), pi_log, X);
                math21_operator_matrix_reverse_x_axis(pi_log);
                pi_log.log("Optimal policy without usable Ace");
                math21_operator_tensor_shrink(V.getTensor(), V_log, X);
                math21_operator_matrix_reverse_x_axis(V_log);
                V_log.log("Optimal value without usable Ace");
            }

            void figure_5_3(ImageDraw &f) {
                NumR true_value = -0.27726;
                NumN episodes = (NumN) (1e04);
                NumN runs = 100;
                VecR eo(episodes);
                VecR ew(episodes);
                eo = 0;
                ew = 0;
                NumN i;
                VecR OS(episodes);
                VecR WS(episodes);
                for (i = 1; i <= runs; ++i) {
                    MC_estimate_state_value_off_policy(OS, WS, episodes);

                    math21_operator_container_subtract_A_k_to(OS, true_value);
                    math21_operator_container_square_to(OS);
                    math21_operator_container_addToA(eo, OS);

                    math21_operator_container_subtract_A_k_to(WS, true_value);
                    math21_operator_container_square_to(WS);
                    math21_operator_container_addToA(ew, WS);
                }
                math21_operator_container_linear_to_A((1.0 / runs), eo);
                math21_operator_container_linear_to_A((1.0 / runs), ew);

                // draw
                VecR S_draw(2, episodes);
                VecN X;
                X.setSize(2);
                X = 1, 0;
                TenSubR tenSub = S_draw.shrinkSub(X);
                tenSub.assign(eo);
                X = 2, 0;
                TenSubR tenSub2 = S_draw.shrinkSub(X);
                tenSub2.assign(ew);
                f.setDir("figure_5_3");
                f.draw(S_draw, "Ordinary_and_Weighted_Importance_Sampling");
//                f.draw(eo, "Ordinary Importance Sampling", 0);
//                f.draw(ew, "Weighted Importance Sampling", 0);
                eo.log("Ordinary Importance Sampling");
                ew.log("Weighted Importance Sampling");
            }

            void test(ImageDraw &f) {
                init();
                figure_5_1();
                figure_5_2();
                figure_5_3(f);
            }
        }
    }

    void math21_rl_chapter05_blackjack(ImageDraw &f) {
        m21log("blackjack");
        rl::blackjack::test(f);
    }
}