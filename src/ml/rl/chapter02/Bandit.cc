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
#include "Bandit.h"

namespace math21 {
    namespace rl {
        DefaultRandomEngine engine(21);
        RanUniform ranUniform(engine);
        RanNormal ranNormal(engine);

        Bandit::Bandit() {
            init();
        }

        // Copy constructor
//                Bandit(const Bandit &B);

        Bandit::~Bandit() {

        }


        void Bandit::init(const BanditParas &paras) {
            this->k = paras.k;
            this->epsilon = paras.epsilon;
            this->initial = paras.initial;
            this->alpha = paras.alpha;
            this->is_using_sample_averages = paras.is_using_sample_averages;
            this->c = paras.c;
            this->is_using_gradient = paras.is_using_gradient;
            this->is_using_gradient_baseline = paras.is_using_gradient_baseline;
            this->true_reward = paras.true_reward;

            As.setSize(k);
            As.letters();
            t = 0;
            R_bar = 0.0;

            q_star.setSize(k);
            Q.setSize(k);
            N.setSize(k);

            if (c != 0) {
                Q_UCB.setSize(k);
            }

            pi.setSize(k);
        }

        // reset is used in algorithm level.
        void Bandit::reset() {
            ranNormal.set(true_reward, 1);
            math21_random_draw(q_star, ranNormal);
            Q = initial;
            N = 0;
            A_star = math21_operator_argmax(q_star);
            t = 0;
        }

        //
        NumN Bandit::act() {
            ranUniform.set(0, 1);
            NumR x;
            math21_random_draw(x, ranUniform);
            if (x < epsilon) {
                return math21_operator_container_arg_random(As, engine);
            }
            if (c != 0) {
                if (t < k) {
                    NumN i = math21_operator_container_arg(N, (NumN) 0);
                    MATH21_ASSERT(i)
                    return i;
                }
                math21_operator_container_divide(xjlog(t), N, Q_UCB);
                math21_operator_container_sqrt_to(Q_UCB);
                math21_operator_container_linear_to_B(1, Q, c, Q_UCB);
                return math21_operator_container_argmax_random(Q_UCB, engine);
            }
            if (is_using_gradient) {
                VecR &H = Q;
                math21_operator_container_exp(H, pi);
                NumR sum = math21_operator_container_sum(pi, 1);
                math21_operator_container_linear_to_A(1 / sum, pi);
                RanDiscrete ranDiscrete(pi, engine);
                NumN i;
                math21_random_draw(i, ranDiscrete);
                return As(i);
            }
            return math21_operator_container_argmax_random(Q, engine);
        }

        NumR Bandit::step(NumN A) {
            NumR R;
            ranNormal.set(q_star(A), 1);
            math21_random_draw(R, ranNormal);
            ++t;
            N(A) = N(A) + 1;
            R_bar = R_bar + (1.0 / t) * (R - R_bar);
            if (is_using_sample_averages) {
                Q(A) = Q(A) + (1.0 / N(A)) * (R - Q(A));
            } else if (is_using_gradient) {
                VecR onehot;
                onehot.setSize(k);
                onehot = 0;
                onehot(A) = 1;
                NumR baseline;
                if (is_using_gradient_baseline) {
                    baseline = R_bar;
                } else {
                    baseline = 0;
                }
                VecR &H = Q;
                math21_operator_container_linear_to_A(1, onehot, -1, pi);
                math21_operator_container_linear_to_A(1, H, alpha * (R - baseline), onehot);
            } else {
                Q(A) = Q(A) + alpha * (R - Q(A));
            }
            return R;
        }

        NumN Bandit::get_best_action() const {
            return A_star;
        }

        void Bandit::log(const char *name) const {
            log(std::cout, name);
        }

        void Bandit::log(std::ostream &io, const char *name) const {
            if (name) {
                io << "Bandit " << name << ":\n";
            }
            MATH21_LOG_NAME_VALUE(k);
            MATH21_LOG_NAME_VALUE(epsilon);
            MATH21_LOG_NAME_VALUE(initial);
            MATH21_LOG_NAME_VALUE(alpha);
            MATH21_LOG_NAME_VALUE(is_using_sample_averages);
            MATH21_LOG_NAME_VALUE(c);
            MATH21_LOG_NAME_VALUE(is_using_gradient);
            MATH21_LOG_NAME_VALUE(is_using_gradient_baseline);
            MATH21_LOG_NAME_VALUE(true_reward);
            As.log("As");
            MATH21_LOG_NAME_VALUE(t);
            MATH21_LOG_NAME_VALUE(R_bar);
            q_star.log("q*");
            Q.log("Q");
            N.log("N");
            MATH21_LOG_NAME_VALUE("A*", A_star);
        }

        void simulate(Seqce<Bandit> &bandits, TenR &mean_best_action_counts, TenR &mean_rewards, NumN runs = 2000,
                      NumN time = 1000) {
            TenR rewards(bandits.size(), runs, time);
            rewards = 0;
            TenN best_action_counts(rewards.shape());
            best_action_counts = 0;
            for (NumN i = 1; i <= bandits.size(); ++i) {
                for (NumN run = 1; run <= runs; ++run) {
                    Bandit &bandit = bandits.at(i);
                    bandit.reset();
                    for (NumN t = 1; t <= time; ++t) {
                        NumN a = bandit.act();
                        NumR r = bandit.step(a);
                        rewards(i, run, t) = r;
                        if (a == bandit.get_best_action()) {
                            best_action_counts(i, run, t) = 1;
                        }
                    }
                }
            }
            TensorFunction_mean f_mean;
            VecN index(best_action_counts.dims());
            index = 0, 1, 0;

            math21_operator_tensor_f_shrink(best_action_counts, mean_best_action_counts, index, f_mean);
            math21_operator_tensor_f_shrink(rewards, mean_rewards, index, f_mean);
        }

        void figure_2_2(ImageDraw &f, NumN runs = 2000, NumN time = 1000) {
            VecR epsilons(3);
            epsilons = 0, 0.1, 0.01;

            Seqce<Bandit> bandits(epsilons.size());
            BanditParas paras;
            for (NumN i = 1; i <= epsilons.size(); ++i) {
                Bandit &bandit = bandits.at(i);
                paras.epsilon = epsilons(i);
                paras.is_using_sample_averages = 1;
                bandit.init(paras);
            }

            TenR mean_best_action_counts;
            TenR mean_rewards;
            simulate(bandits, mean_best_action_counts, mean_rewards, runs, time);

            mean_best_action_counts.log("mean_best_action_counts");
            mean_rewards.log("mean_rewards");

            f.setDir("figure_2_2");
            f.draw(mean_best_action_counts, "optimal_action");
            f.draw(mean_rewards, "average_reward");
        }

        void figure_2_3(ImageDraw &f, NumN runs = 2000, NumN time = 1000) {
            Seqce<BanditParas> paras_batch(2);
            Seqce<Bandit> bandits(paras_batch.size());
            BanditParas &paras = paras_batch.at(1);
            paras.epsilon = 0;
            paras.initial = 5;
            paras.alpha = 0.1;
            BanditParas &paras2 = paras_batch.at(2);
            paras2.epsilon = 0.1;
            paras2.initial = 0;
            paras2.alpha = 0.1;

            for (NumN i = 1; i <= paras_batch.size(); ++i) {
                Bandit &bandit = bandits.at(i);
                bandit.init(paras_batch(i));
            }

            TenR mean_best_action_counts;
            TenR mean_rewards;
            simulate(bandits, mean_best_action_counts, mean_rewards, runs, time);

            mean_best_action_counts.log("mean_best_action_counts");
//            mean_rewards.log("mean_rewards");

            f.setDir("figure_2_3");
            f.draw(mean_best_action_counts, "optimal_action");
        }

        void figure_2_4(ImageDraw &f, NumN runs = 2000, NumN time = 1000) {
            Seqce<BanditParas> paras_batch(2);
            Seqce<Bandit> bandits(paras_batch.size());
            BanditParas &paras = paras_batch.at(1);
            paras.epsilon = 0;
            paras.c = 2;
            paras.is_using_sample_averages = 1;
            BanditParas &paras2 = paras_batch.at(2);
            paras2.epsilon = 0.1;
            paras2.is_using_sample_averages = 1;

            for (NumN i = 1; i <= paras_batch.size(); ++i) {
                Bandit &bandit = bandits.at(i);
                bandit.init(paras_batch(i));
            }

            TenR mean_best_action_counts;
            TenR mean_rewards;
            simulate(bandits, mean_best_action_counts, mean_rewards, runs, time);

//            mean_best_action_counts.log("mean_best_action_counts");
            mean_rewards.log("mean_rewards");

            f.setDir("figure_2_4");
            f.draw(mean_rewards, "average_reward");
        }

        void figure_2_5(ImageDraw &f, NumN runs = 2000, NumN time = 1000) {
            Seqce<BanditParas> paras_batch(4);
            Seqce<Bandit> bandits(paras_batch.size());
            BanditParas &paras = paras_batch.at(1);
            paras.is_using_gradient = 1;
            paras.alpha = 0.1;
            paras.is_using_gradient_baseline = 1;
            paras.true_reward = 4;
            BanditParas &paras2 = paras_batch.at(2);
            paras2.is_using_gradient = 1;
            paras2.alpha = 0.1;
            paras2.is_using_gradient_baseline = 0;
            paras2.true_reward = 4;
            BanditParas &paras3 = paras_batch.at(3);
            paras3.is_using_gradient = 1;
            paras3.alpha = 0.4;
            paras3.is_using_gradient_baseline = 1;
            paras3.true_reward = 4;
            BanditParas &paras4 = paras_batch.at(4);
            paras4.is_using_gradient = 1;
            paras4.alpha = 0.4;
            paras4.is_using_gradient_baseline = 0;
            paras4.true_reward = 4;
            for (NumN i = 1; i <= paras_batch.size(); ++i) {
                Bandit &bandit = bandits.at(i);
                bandit.init(paras_batch(i));
            }

            TenR mean_best_action_counts;
            TenR mean_rewards;
            simulate(bandits, mean_best_action_counts, mean_rewards, runs, time);

            mean_best_action_counts.log("mean_best_action_counts");
//            mean_rewards.log("mean_rewards");

            f.setDir("figure_2_5");
            f.draw(mean_best_action_counts, "optimal_action");
        }

        void figure_2_6(ImageDraw &f, NumN runs = 2000, NumN time = 1000) {
            Seqce<VecZ> paras_batch(4);
            paras_batch.at(1).setSize(6);
            math21_operator_container_letters(paras_batch.at(1), -7);
            paras_batch.at(2).setSize(7);
            math21_operator_container_letters(paras_batch.at(2), -5);
            paras_batch.at(3).setSize(7);
            math21_operator_container_letters(paras_batch.at(3), -4);
            paras_batch.at(4).setSize(5);
            math21_operator_container_letters(paras_batch.at(4), -2);

            NumN n = math21_operator_container_2d_size(paras_batch);
            Seqce<Bandit> bandits(n);

            NumN k = 1;
            for (NumN i = 1; i <= paras_batch.size(); ++i) {
                const VecZ &ps = paras_batch(i);
                if (i == 1) {
                    for (NumN j = 1; j <= ps.size(); ++j) {
                        BanditParas paras;
                        paras.epsilon = m21_pow(2, ps(j));
                        paras.is_using_sample_averages = 1;
                        Bandit &bandit = bandits.at(k);
                        bandit.init(paras);
                        ++k;
                    }
                } else if (i == 2) {
                    for (NumN j = 1; j <= ps.size(); ++j) {
                        BanditParas paras;
                        paras.alpha = m21_pow(2, ps(j));
                        paras.is_using_gradient = 1;
                        paras.is_using_gradient_baseline = 1;
                        Bandit &bandit = bandits.at(k);
                        bandit.init(paras);
                        ++k;
                    }
                } else if (i == 3) {
                    for (NumN j = 1; j <= ps.size(); ++j) {
                        BanditParas paras;
                        paras.epsilon = 0;
                        paras.c = m21_pow(2, ps(j));
                        paras.is_using_sample_averages = 1;
                        Bandit &bandit = bandits.at(k);
                        bandit.init(paras);
                        ++k;
                    }
                } else if (i == 4) {
                    for (NumN j = 1; j <= ps.size(); ++j) {
                        BanditParas paras;
                        paras.epsilon = 0;
                        paras.initial = m21_pow(2, ps(j));
                        paras.alpha = 0.1;
                        Bandit &bandit = bandits.at(k);
                        bandit.init(paras);
                        ++k;
                    }
                }
            }

            TenR mean_best_action_counts;
            TenR mean_rewards;
            simulate(bandits, mean_best_action_counts, mean_rewards, runs, time);

            TensorFunction_mean f_mean;
            VecN index(mean_rewards.dims());
            index = 0, 1;

            TenR rewards;
            math21_operator_tensor_f_shrink(mean_rewards, rewards, index, f_mean);

            rewards.log("rewards");

            // draw
            NumN ele_max = math21_operator_container_2d_element_size_max(paras_batch);
            TenR rewards_draw(paras_batch.size(), ele_max, 2);

            k = 1;
            for (NumN i = 1; i <= paras_batch.size(); ++i) {
                const VecZ &ps = paras_batch(i);
                for (NumN j = 1; j <= ele_max; ++j) {
                    if (j <= ps.size()) {
                        rewards_draw(i, j, 1) = ps(j);
                        rewards_draw(i, j, 2) = rewards(k);
                        ++k;
                    } else {
                        rewards_draw(i, j, 1) = rewards_draw(i, 1, 1);
                        rewards_draw(i, j, 2) = rewards_draw(i, 1, 2);
                    }
                }
            }

            f.setDir("figure_2_6");
            f.draw(rewards_draw, "average_reward");
        }

        void rl_k_armed_bandit_test(ImageDraw &f, NumN figure_num, NumN runs = 2000, NumN time = 1000) {
            m21vlog("figure_num = %d, runs = %d, time = %d\n", figure_num, runs, time);
//            ImageDraw_Dummy f0;
//            figure_2_2(f0);

            if (figure_num == 2) {
                figure_2_2(f, runs, time);
            } else if (figure_num == 3) {
                figure_2_3(f, runs, time);
            } else if (figure_num == 4) {
                figure_2_4(f, runs, time);
            } else if (figure_num == 5) {
                figure_2_5(f, runs, time);
            } else if (figure_num == 6) {
                figure_2_6(f, runs, time);
            } else if (figure_num == 0) {
                figure_2_2(f, runs, time);
                figure_2_3(f, runs, time);
                figure_2_4(f, runs, time);
                figure_2_5(f, runs, time);
                figure_2_6(f, runs, time);
            }

        }
    }

    void math21_rl_chapter02_k_armed_bandit(ImageDraw &f, NumN figure_num, NumN runs, NumN time) {
        rl::rl_k_armed_bandit_test(f, figure_num, runs, time);
    }

}