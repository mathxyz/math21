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

#include "inner.h"

namespace math21 {

    void generate_sin_sample(MatR &X, MatR &Y, MatR &FX, MatR &FY,
                             NumR f = 1, NumR *pt0 = 0, NumN batch_size = 1, NumN n_samples = 100, NumN n_predict = 50);

    namespace ad {
        ad_point ad_rnn_sin01_predict(ad_point x, const ad_point &theta_rnn, const ad_point &w, ad_point b,
                                      NumN n_input, NumN n_steps, NumN n_hidden, NumN n_outputs);
    }

    struct m21lstm_sin01_type_config {
        NumB predict;

        std::string textPath;
        std::string functionParasPath;
        std::string functionParasPath_init; // can be empty
        std::string functionParasPath_save; // can be empty

        // display
        NumN display_step;

        // data
        NumN time_steps;
        NumN max_batches;
        NumN batch_size;
        NumN alphabet_size;

        // f
        NumN input_size;
        NumN state_size;
        NumN output_size;
        NumN deviceType;

        // optimization
        NumN num_iters;
        NumR step_size;
        NumB printTextInOpt;

        // generate text
        NumN n_lines;
        NumN sequence_length;

        NumB debug;

        m21lstm_sin01_type_config();

        void log(const char *name = 0) const;
    };

    void math21_ml_lstm_sin01(const m21lstm_sin01_type_config &config);
}