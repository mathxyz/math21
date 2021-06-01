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

#pragma once

#include "inner_header.h"

namespace math21 {
    namespace rl {
        struct BanditParas {
        private:
            void init(NumN k = 10, NumR epsilon = 0.0, NumR initial = 0.0, NumR alpha = 0.1,
                      NumB is_using_sample_averages = 0, NumR c = 0, NumB is_using_gradient = 0,
                      NumB is_using_gradient_baseline = 0, NumR true_reward = 0.0) {
                this->k = k;
                this->epsilon = epsilon;
                this->initial = initial;
                this->alpha = alpha;
                this->is_using_sample_averages = is_using_sample_averages;
                this->c = c;
                this->is_using_gradient = is_using_gradient;
                this->is_using_gradient_baseline = is_using_gradient_baseline;
                this->true_reward = true_reward;
            }

        public:
            NumN k;
            NumR epsilon;
            NumR initial;
            NumR alpha;
            NumB is_using_sample_averages;
            NumR c;
            NumB is_using_gradient;
            NumB is_using_gradient_baseline;
            NumR true_reward;

            BanditParas() {
                init();
            }

            void reset() {
                init();
            }
        };

        struct Bandit {
        private:
            NumN k;
            NumR epsilon;
            NumR initial;
            NumR alpha;
            NumB is_using_sample_averages;
            NumR c;
            NumB is_using_gradient;
            NumB is_using_gradient_baseline;
            NumR true_reward;

            VecN As;
            NumN t;
            NumR R_bar;

            VecR q_star;
            VecR Q;
            VecN N;
            NumN A_star;

            VecR Q_UCB; // tmp
            VecR pi;
        public:

//                void copyTo(Bandit &B) const;

            Bandit();

            // Copy constructor
//                Bandit(const Bandit &B);

            virtual ~Bandit();

            void init(const BanditParas &paras = BanditParas());

            // reset is used in algorithm level.
            void reset();

            NumN act();

            NumR step(NumN A);

            NumN get_best_action() const;

            void log(const char *name = 0) const;

            void log(std::ostream &io, const char *name = 0) const;
//
//                NumZ getWinner() const;
//
//                void serialize(std::ostream &io, SerializeNumInterface &sn) const;
//
//                void deserialize(std::istream &io, DeserializeNumInterface &sn);

        };
    }

    void math21_rl_chapter02_k_armed_bandit(ImageDraw &f, NumN figure_num, NumN runs = 2000, NumN time = 1000);
}