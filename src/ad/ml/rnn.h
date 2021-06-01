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
    namespace ad {
        namespace rnn_detail {
            struct ad_rnn_params {
                ad_point params_1d, init_hiddens, change_para, predict_para;
            };

            struct ad_lstm_params {
                ad_point params_1d, init_hiddens, change_para, predict_para,
                        init_cells, forget_para, ingate_para, outgate_para;
            };

            void split_rnn_params(const ad_point &params_point, ad_rnn_params &params,
                                  NumN input_size, NumN state_size, NumN output_size);

            void split_lstm_params(const ad_point &params_point, ad_lstm_params &params,
                                   NumN input_size, NumN state_size, NumN output_size);

            ad_point concat_and_multiply(const ad_point &weights, const Seqce<ad_point> &args);

            void updata_rnn(const ad_point &input, ad_point &hiddens, const ad_rnn_params &params);

            void update_lstm(const ad_point &input, ad_point &hiddens, ad_point &cells, const ad_lstm_params &params);

            ad_point hiddens_to_output_probs(const ad_point &hiddens, const ad_point &predict_para);
        }

        typedef struct {
            ad_point x;
            ad_point y;
        } lstm_f_pair_args;

        typedef struct {
            NumB printText;
            ad_point x;
            ad_point y;
            ad_point logprobs;
            ad_point data_targets;
        } lstm_f_callback_args;

        NumN ad_rnn_calculate_params_size(NumN input_size, NumN state_size, NumN output_size);

        NumN ad_lstm_calculate_params_size(NumN input_size, NumN state_size, NumN output_size);

        void ad_rnn_init_params(
                const ad_point &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale = 0.01);

        void ad_lstm_init_params(
                const ad_point &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale = 0.01);

        ad_point ad_rnn_predict(const ad_point &params_point, const ad_point &data_inputs,
                                NumN input_size, NumN state_size, NumN output_size);

        ad_point ad_lstm_predict(const ad_point &params_point, const ad_point &data_inputs,
                                 NumN input_size, NumN state_size, NumN output_size);

        void ad_lstm_only_hiddens(const ad_point &params_point, const ad_point &data_inputs,
                               Seqce <ad_point> &outputs,
                               NumN input_size, NumN state_size, NumN output_size);

        ad_point ad_rnn_log_likelihood(
                const ad_point &params_point, const ad_point &data_inputs, const ad_point &data_targets,
                NumN input_size, NumN state_size, NumN output_size);

        ad_point ad_rnn_part_log_likelihood(const ad_point &logprobs, const ad_point &data_targets);
    }

    void math21_data_text_lstm_generate_text(const TenR &x_value_final,
                                             NumN input_size,
                                             NumN state_size,
                                             NumN output_size,
                                             NumN n_lines,
                                             NumN sequence_length,
                                             NumN alphabet_size,
                                             NumN deviceType = m21_device_type_default);

    void math21_data_text_lstm_print_training_prediction(const TenR &x_cur, void *data);

    struct m21lstm_text_type_config {
        NumB predict;

        std::string textPath;
        std::string functionParasPath;
        std::string functionParasPath_init; // can be empty
        std::string functionParasPath_save; // can be empty

        // data
        NumN max_batches;
        NumN alphabet_size;

        // f
        NumN input_size;
        NumN state_size;
        NumN output_size;
        NumN time_steps;
        NumN deviceType;

        // optimization
        NumN num_iters;
        NumR step_size;
        NumB printTextInOpt;

        // generate text
        NumN n_lines;
        NumN sequence_length;

        NumB debug;

        m21lstm_text_type_config();

        void log(const char *name = 0) const;
    };

    // Create a graph using text, code, or graph. They are all equivalent.
    // Here graph is created from code.
    void math21_ml_lstm_text(const m21lstm_text_type_config &config);
}