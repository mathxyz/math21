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

#include "../../point/files.h"
#include "../../tool/files.h"
#include "../../op/files.h"
#include "../../opt/files.h"
#include "rnn.h"

namespace math21 {
    namespace ad {
        namespace rnn_detail {
            void split_rnn_params(const ad_point &params_point, ad_rnn_params &params,
                                  NumN input_size, NumN state_size, NumN output_size) {
                ad_point &params_1d = params.params_1d;
                ad_point &init_hiddens = params.init_hiddens;
                ad_point &change_para = params.change_para;
                ad_point &predict_para = params.predict_para;

                NumN h_size = 1 * state_size;
                NumN wc_size = (input_size + state_size + 1) * state_size;
                NumN wp_size = (state_size + 1) * output_size;

                params_1d = params_point;

                init_hiddens = ad_vec_share_sub_from_to(params_1d, 1, h_size);
                change_para = ad_vec_share_sub_from_to(params_1d, h_size + 1, h_size + wc_size);
                predict_para = ad_vec_share_sub_from_to(params_1d, h_size + wc_size + 1, -1);
                VecN d(2);
                d = 1, state_size;
                init_hiddens = ad_share_reshape(init_hiddens, d);
                d = input_size + state_size + 1, state_size;
                change_para = ad_share_reshape(change_para, d);
                d = state_size + 1, output_size;
                predict_para = ad_share_reshape(predict_para, d);
            }

            void split_lstm_params(const ad_point &params_point, ad_lstm_params &params,
                                   NumN input_size, NumN state_size, NumN output_size) {
                ad_point &params_1d = params.params_1d;
                ad_point &init_hiddens = params.init_hiddens;
                ad_point &change_para = params.change_para;
                ad_point &predict_para = params.predict_para;
                ad_point &init_cells = params.init_cells;
                ad_point &forget_para = params.forget_para;
                ad_point &ingate_para = params.ingate_para;
                ad_point &outgate_para = params.outgate_para;

                NumN h_size = 1 * state_size;
                NumN wc_size = (input_size + state_size + 1) * state_size;
                NumN wp_size = (state_size + 1) * output_size;
                NumN cell_size = h_size;
                NumN forget_size = wc_size;
                NumN ingate_size = wc_size;
                NumN outgate_size = wc_size;

                params_1d = params_point;

                NumN offset = 0;
                init_hiddens = ad_vec_share_sub_offset(params_1d, offset, h_size);
                offset += h_size;
                change_para = ad_vec_share_sub_offset(params_1d, offset, wc_size);
                offset += wc_size;
                predict_para = ad_vec_share_sub_offset(params_1d, offset, wp_size);
                offset += wp_size;
                init_cells = ad_vec_share_sub_offset(params_1d, offset, cell_size);
                offset += cell_size;
                forget_para = ad_vec_share_sub_offset(params_1d, offset, forget_size);
                offset += forget_size;
                ingate_para = ad_vec_share_sub_offset(params_1d, offset, ingate_size);
                offset += ingate_size;
                outgate_para = ad_vec_share_sub_offset(params_1d, offset, outgate_size);
                VecN d(2);
                d = 1, state_size;
                init_hiddens = ad_share_reshape(init_hiddens, d);
                d = input_size + state_size + 1, state_size;
                change_para = ad_share_reshape(change_para, d);
                d = state_size + 1, output_size;
                predict_para = ad_share_reshape(predict_para, d);
                d = 1, state_size;
                init_cells = ad_share_reshape(init_cells, d);
                d = input_size + state_size + 1, state_size;
                forget_para = ad_share_reshape(forget_para, d);
                d = input_size + state_size + 1, state_size;
                ingate_para = ad_share_reshape(ingate_para, d);
                d = input_size + state_size + 1, state_size;
                outgate_para = ad_share_reshape(outgate_para, d);
            }

            ad_point concat_and_multiply(const ad_point &weights, const Seqce<ad_point> &args) {
                Seqce<ad_point> xs(args.size() + 1);
                math21_operator_container_set_partially(args, xs, 0, 0, args.size());
                NumN nr = ad_get_dim_i(args(1), 1);
                MatR k_value(nr, 1);
                k_value = 1;
                xs(xs.size()) = ad_point(k_value, 0, ad_get_device_type(xs(1)));
                auto cat_state = ad_concatenate(xs, 2);
                auto result = ad_mat_mul(cat_state, weights);
                return result;
            }

            // x: num_sequences, input_size
// h: num_sequences, state_size
// 1: num_sequences, 1
// wx: input_size, state_size
// wh: state_size, state_size
// b: 1, state_size
// (x, h, 1): num_sequences, (input_size + state_size +1)
// paras: input_size + state_size + 1, state_size
// h = x*wx + h*wh + b
//   = (x, h, 1) * (wx,
//                  wh,
//                  b)
            void updata_rnn(const ad_point &input, ad_point &hiddens, const ad_rnn_params &params) {
                Seqce<ad_point> args;
                args.push(input);
                args.push(hiddens);
                hiddens = ad_tanh(concat_and_multiply(params.change_para, args));
            }

            // See 'class BasicLSTMCell(LayerRNNCell)' in tf.
            // input, hiddens, cells -> hiddens, cells
            void update_lstm(const ad_point &input, ad_point &hiddens, ad_point &cells, const ad_lstm_params &params) {
                Seqce<ad_point> args;
                args.push(input);
                args.push(hiddens);
                // todo: Parameters of gates are concatenated into one multiply for efficiency.
                auto change = ad_tanh(concat_and_multiply(params.change_para, args));
                auto forget = ad_sigmoid_from_tanh(concat_and_multiply(params.forget_para, args));
                auto ingate = ad_sigmoid_from_tanh(concat_and_multiply(params.ingate_para, args));
                auto outgate = ad_sigmoid_from_tanh(concat_and_multiply(params.outgate_para, args));
                cells = cells * forget + ingate * change;
                hiddens = outgate * ad_tanh(cells);
            }

// hiddens: num_sequences * state_size
// paras: state_size + 1, output_size
// output: num_sequences * output_size
            ad_point hiddens_to_output_probs(const ad_point &hiddens, const ad_point &predict_para) {
                Seqce<ad_point> args;
                args.push(hiddens);
                auto output = concat_and_multiply(predict_para, args);
                VecN axes(1);
                axes = 2;
                auto logprob = ad_logsumexp(output, axes, 1); // Normalize log-probs
                auto result = output - logprob;
                return result;
            }
        }

        using namespace rnn_detail;

        NumN ad_rnn_calculate_params_size(NumN input_size, NumN state_size, NumN output_size) {
            NumN h_size = 1 * state_size;
            NumN wc_size = (input_size + state_size + 1) * state_size;
            NumN wp_size = (state_size + 1) * output_size;
            return h_size + wc_size + wp_size;
        }

        NumN ad_lstm_calculate_params_size(NumN input_size, NumN state_size, NumN output_size) {
            NumN h_size = 1 * state_size;
            NumN wc_size = (input_size + state_size + 1) * state_size;
            NumN wp_size = (state_size + 1) * output_size;
            NumN cell_size = h_size;
            NumN forget_size = wc_size;
            NumN ingate_size = wc_size;
            NumN outgate_size = wc_size;
            return h_size + wc_size + wp_size + cell_size + forget_size + ingate_size + outgate_size;
        }

        void ad_rnn_init_params(
                const ad_point &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale) {
            ad_rnn_params params;
            split_rnn_params(params_point, params, input_size, state_size, output_size);
            RanNormal ranNormal;
            ranNormal.set(0, param_scale);
            math21_op_random_draw(ad_get_value(params.init_hiddens), ranNormal);
            math21_op_random_draw(ad_get_value(params.change_para), ranNormal);
            math21_op_random_draw(ad_get_value(params.predict_para), ranNormal);
        }

        void ad_lstm_init_params(
                const ad_point &params_point,
                NumN input_size, NumN state_size, NumN output_size, NumR param_scale) {
            ad_lstm_params params;
            split_lstm_params(params_point, params, input_size, state_size, output_size);

            DefaultRandomEngine engine(21);
            RanNormal ranNormal(engine);
            ranNormal.set(0, param_scale);
            math21_op_random_draw(ad_get_value(params.init_hiddens), ranNormal);
            math21_op_random_draw(ad_get_value(params.change_para), ranNormal);
            math21_op_random_draw(ad_get_value(params.predict_para), ranNormal);
            math21_op_random_draw(ad_get_value(params.init_cells), ranNormal);
            math21_op_random_draw(ad_get_value(params.forget_para), ranNormal);
            math21_op_random_draw(ad_get_value(params.ingate_para), ranNormal);
            math21_op_random_draw(ad_get_value(params.outgate_para), ranNormal);
        }

        ad_point ad_rnn_predict(const ad_point &params_point, const ad_point &data_inputs,
                                NumN input_size, NumN state_size, NumN output_size) {
            ad_rnn_params params;
            split_rnn_params(params_point, params, input_size, state_size, output_size);

            NumN num_sequences = yf_get_dim_i(data_inputs, 2);
            // hiddens: num_sequences, state_size
            auto hiddens = ad_repeat(params.init_hiddens, num_sequences, 1);
            auto output = hiddens_to_output_probs(hiddens, params.predict_para);
            _Set<ad_point> pack_input;
            pack_input.add(output);
            if (yf_get_dim_i(data_inputs, 1) > 0) {
                _Set<ad_point> input_s;
                ad_row_unpack(data_inputs, input_s);
                for (NumN i = 1; i <= input_s.size(); ++i) { // Iterate over time steps.
                    auto input = input_s(i);
                    updata_rnn(input, hiddens, params);
                    output = hiddens_to_output_probs(hiddens, params.predict_para);
                    pack_input.add(output);
                }
            }
            // output: time_steps * num_sequences * output_size
            output = ad_row_pack(pack_input);
            return output;
        }

        // See 'def static_rnn' in tf.
        // input_size = feature_size
        ad_point ad_lstm_predict(const ad_point &params_point, const ad_point &data_inputs,
                                 NumN input_size, NumN state_size, NumN output_size) {
            MATH21_ASSERT(input_size == yf_get_dim_i(data_inputs, 3), "input_size = feature_size not met!");
            ad_lstm_params params;
            split_lstm_params(params_point, params, input_size, state_size, output_size);

            NumN num_sequences = yf_get_dim_i(data_inputs, 2);
            // hiddens: num_sequences, state_size
            auto hiddens = ad_repeat(params.init_hiddens, num_sequences, 1);
            auto cells = ad_repeat(params.init_cells, num_sequences, 1);
            auto output = hiddens_to_output_probs(hiddens, params.predict_para);
            _Set<ad_point> pack_input;
            pack_input.add(output);
            if (yf_get_dim_i(data_inputs, 1) > 0) {
                _Set<ad_point> input_s;
                ad_row_unpack(data_inputs, input_s);
                for (NumN i = 1; i <= input_s.size(); ++i) { // Iterate over time steps.
                    auto input = input_s(i);
                    update_lstm(input, hiddens, cells, params);
                    output = hiddens_to_output_probs(hiddens, params.predict_para);
                    pack_input.add(output);
                }
            } else { // dims of data_inputs may not be zero even if it is empty.
            }
            // output: time_steps * num_sequences * output_size
            output = ad_row_pack(pack_input);
            return output;
        }

        // See 'def static_rnn' in tf.
        // outputs: time_steps * [num_sequences * state_size]
        // input_size = feature_size
        void ad_lstm_only_hiddens(const ad_point &params_point, const ad_point &data_inputs,
                                  Seqce<ad_point> &outputs,
                                  NumN input_size, NumN state_size, NumN output_size) {
            MATH21_ASSERT(input_size == yf_get_dim_i(data_inputs, 3), "input_size = feature_size not met!");
            ad_lstm_params params;
            split_lstm_params(params_point, params, input_size, state_size, output_size);

            NumN num_sequences = yf_get_dim_i(data_inputs, 2);
            // hiddens: num_sequences, state_size
            auto hiddens = ad_repeat(params.init_hiddens, num_sequences, 1);
            auto cells = ad_repeat(params.init_cells, num_sequences, 1);
            outputs.clear();
            if (yf_get_dim_i(data_inputs, 1) > 0) {
                _Set<ad_point> input_s;
                ad_row_unpack(data_inputs, input_s);
                for (NumN i = 1; i <= input_s.size(); ++i) { // Iterate over time steps.
                    auto input = input_s(i);
                    update_lstm(input, hiddens, cells, params);
                    outputs.push(hiddens);
                }
            }
        }

        ad_point ad_rnn_log_likelihood(
                const ad_point &params_point, const ad_point &data_inputs, const ad_point &data_targets,
                NumN input_size, NumN state_size, NumN output_size) {
            auto logprobs = ad_rnn_predict(params_point, data_inputs, input_size, state_size, output_size);
            return ad_rnn_part_log_likelihood(logprobs, data_targets);
        }

        // input_size = output_size
        ad_point ad_rnn_part_log_likelihood(const ad_point &logprobs0, const ad_point &data_targets) {
            NumN num_time_steps = ad_get_dim_i(data_targets, 1);
            NumN num_examples = ad_get_dim_i(data_targets, 2);
            NumN logprobs_dim1 = ad_get_dim_i(logprobs0, 1);

            // not include the last time step
            auto logprobs = ad_axis_i_sub_get(logprobs0, 0, logprobs_dim1 - 1, 1);
            auto loglik = ad_inner_product(logprobs, data_targets);
            NumR denominator = num_time_steps * num_examples;
            MATH21_ASSERT(denominator)
            return loglik / denominator;
        }

        void f_main() {
            NumN input_size = 128;
            NumN state_size = 40;
            NumN output_size = 128;
            NumR param_scale = 0.01;
            NumN n_paras = ad_rnn_calculate_params_size(input_size, state_size, output_size);
            VecR params_value(n_paras);
            ad_point params_point(params_value, 1);
            ad_rnn_init_params(params_point, input_size, state_size, output_size, param_scale);
            ad_point data_inputs, data_targets;
            ad_rnn_log_likelihood(params_point, data_inputs, data_targets,
                                  input_size, state_size, output_size);
        }
    }

    using namespace ad;

    void string_to_one_hot(const VecR &line, const MatR &letters, MatR &oneHotMatrix) {
        math21_op_ele_is_equal(line, letters, oneHotMatrix); // string to one hot
    }

    void one_hot_to_string(const MatR &oneHotMatrix, VecN8 &s) {
        s.setSize(oneHotMatrix.nrows());
        VecR v;
        for (NumN i = 1; i <= oneHotMatrix.nrows(); ++i) {
            math21_op_matrix_get_row(v, oneHotMatrix, i);
            //math21_operator_container_argmax_random()
            NumN c = math21_op_vector_argmax(v);
            s(i) = c - 1;
        }
    }

    void math21_data_text_lstm_generate_text(const TenR &x_value_final,
                                             NumN input_size,
                                             NumN state_size,
                                             NumN output_size,
                                             NumN n_lines,
                                             NumN sequence_length,
                                             NumN alphabet_size,
                                             NumN deviceType) {
        MatR letters(1, alphabet_size);
        letters.letters(0);
        MatR oneHotMatrix;
        for (NumN i = 1; i <= n_lines; ++i) {
            std::string text;
            for (NumN j = 1; j <= sequence_length; ++j) {
                TenN8 line_c;
                math21_operator_tensor_from_string(line_c, (NumN) text.size(), (const NumN8 *) text.c_str());
                VecR line;
                math21_op_vector_set_by_vector(line_c, line);
                string_to_one_hot(line, letters, oneHotMatrix);
                TenR data;
                math21_operator_tensor_share_add_axis(oneHotMatrix, data, 2);

                ad_clear_graph();
                auto data_inputs = ad_point(data, 0, deviceType); // data may have shape (0, 1, 128)
                auto params = ad_point(x_value_final, 0, deviceType);
                auto _ = ad_lstm_predict(params, data_inputs, input_size, state_size, output_size);
                auto &logprobs = ad_get_value(_);

                TenR probs;
                math21_op_tensor_sub_axis_i_get_and_shrink(probs, logprobs, logprobs.dim(1) - 1, 1);
                math21_op_exp_onto(probs);

                if (!probs.is_cpu()) {
                    probs.convertDeviceType(m21_device_type_default);
                }
                RanDiscrete ranDiscrete;
                ranDiscrete.set(probs);
                NumN c_value;
                math21_random_draw(c_value, ranDiscrete);
                auto c = (NumN8) (c_value - 1);
                text += c;
                printf("%c", c);
            }
            printf("\n");
//        m21log(text);
        }
    }

    void math21_data_text_lstm_print_training_prediction(const TenR &x_cur, void *data) {
        // data_targets
        auto args = (lstm_f_callback_args *) data;
        if (!args->printText)return;
        ad_point px = args->x;
        ad_point plogprobs = args->logprobs;
        ad_point pdata_targets = args->data_targets;
        ad_get_value(px) = x_cur;
        ad_fv(plogprobs);
        auto &logprobs = ad_get_value(plogprobs);
        auto &data_targets = ad_get_value(pdata_targets);
        m21log("Training text                         Predicted text");
        for (NumN i = 1; i <= logprobs.dim(2); ++i) {
            TenR training_text_oh, predicted_text_oh;
            VecN8 training_text, predicted_text;
            math21_op_tensor_sub_axis_i_get_and_shrink(training_text_oh, data_targets, i - 1, 2);
            math21_op_tensor_sub_axis_i_get_and_shrink(predicted_text_oh, logprobs, i - 1, 2);
            one_hot_to_string(training_text_oh, training_text);
            one_hot_to_string(predicted_text_oh, predicted_text);
            // remove '\n'
            math21_string_replace(training_text, '\n', ' ');
            math21_string_replace(predicted_text, '\n', ' ');
            std::string s1, s2;
            math21_operator_tensor_to_string(training_text, s1);
            math21_operator_tensor_to_string(predicted_text, s2);
            m21log(s1.append("|").append(s2));
        }
    }

    m21lstm_text_type_config::m21lstm_text_type_config() {
        predict = 0;

        max_batches = 0;
        alphabet_size = 128;

        input_size = 128;
        state_size = 40;
        output_size = 128;
        time_steps = 5;
        deviceType = m21_device_type_default;

        num_iters = 300;
        step_size = 0.1;
        printTextInOpt = 0;

        n_lines = 20;
        sequence_length = 30;

        debug = 0;
    }

    void m21lstm_text_type_config::log(const char *name) const {
        if (name)m21log(name);
#define MATH21_LOCAL_F(a) m21log(MATH21_STRINGIFY(a), a)
        MATH21_LOCAL_F(predict);

        MATH21_LOCAL_F(textPath);
        MATH21_LOCAL_F(functionParasPath);
        MATH21_LOCAL_F(functionParasPath_init);
        MATH21_LOCAL_F(functionParasPath_save);

        MATH21_LOCAL_F(max_batches);
        MATH21_LOCAL_F(alphabet_size);

        MATH21_LOCAL_F(input_size);
        MATH21_LOCAL_F(state_size);
        MATH21_LOCAL_F(output_size);
        MATH21_LOCAL_F(time_steps);
        MATH21_LOCAL_F(deviceType);

        MATH21_LOCAL_F(num_iters);
        MATH21_LOCAL_F(step_size);
        MATH21_LOCAL_F(printTextInOpt);

        MATH21_LOCAL_F(n_lines);
        MATH21_LOCAL_F(sequence_length);
#undef MATH21_LOCAL_F
    }

    static void f_1d(const TenR &x, TenR &y, NumN iter, void *data) {
        auto args = (lstm_f_callback_args *) data;
        ad_point px = args->x;
        ad_point py = args->y;
        ad_get_value(px) = x;
        ad_fv(py);
        y = ad_get_value(py);
        math21_op_vector_kx_onto(-1, y);
    }

    static void df_1d(const TenR &x, TenR &dy, NumN iter, void *data) {
        auto args = (lstm_f_pair_args *) data;
        ad_point px = args->x;
        ad_point pdy = args->y;
        ad_get_value(px) = x;
        ad_fv(pdy);
        dy = ad_get_value(pdy);
        math21_op_vector_kx_onto(-1, dy);
    }

    static void callback(const TenR &x_cur, const TenR &gradient, NumN iter, void *data) {
        if (iter % 10 == 1) {
            TenR y;
            f_1d(x_cur, y, iter, data);
            printf("Iteration %d Train loss: %.16lf\n", iter - 1, y(1)); // use below
            std::flush(std::cout);
//        printf("Iteration %d Train loss: %lf\n", iter, y(1));
            math21_data_text_lstm_print_training_prediction(x_cur, data);
        }
    }

    void math21_ml_lstm_text(const m21lstm_text_type_config &config) {

        ad_clear_graph();

        NumR param_scale = 0.01;

        if (config.predict) {
            TenR x_value_final;
            math21_io_load(config.functionParasPath.c_str(), x_value_final);
            math21_data_text_lstm_generate_text(x_value_final, config.input_size, config.state_size, config.output_size,
                                                config.n_lines,
                                                config.sequence_length,
                                                config.alphabet_size,
                                                config.deviceType);
            return;
        }

        std::string textPath;
        if (config.debug) {
            textPath = math21_string_to_string(MATH21_INCLUDE_PATH) + "/../LICENSE";
        } else {
            textPath = config.textPath;
        }

        // prepare data
        TenR data;
        math21_data_text_build_dataset(textPath.c_str(), data, config.time_steps, config.max_batches,
                                       config.alphabet_size);

        ad_point data_inputs(data, 0, config.deviceType);
        ad_point data_targets(data, 0, config.deviceType);

//        NumN n_paras = ad_rnn_calculate_params_size(config.input_size, config.state_size, config.output_size);
        NumN n_paras = ad_lstm_calculate_params_size(config.input_size, config.state_size, config.output_size);
        VecR params_value(n_paras);
        ad_point params(params_value, 1, ad_get_device_type(data_inputs));

        // init params
//      ad_rnn_init_params(params, config.input_size, config.state_size, config.output_size, param_scale);
        ad_lstm_init_params(params, config.input_size, config.state_size, config.output_size, param_scale);
        if (!config.functionParasPath_init.empty()) {
            math21_io_load(config.functionParasPath_init.c_str(), ad_get_value(params));
        }

//        auto logprobs = ad_rnn_predict(params, data_inputs, config.input_size, config.state_size, config.output_size);
        auto logprobs = ad_lstm_predict(params, data_inputs, config.input_size, config.state_size, config.output_size);

        auto y = ad_rnn_part_log_likelihood(logprobs, data_targets);
//    auto y = ad_rnn_log_likelihood(params, data_inputs, data_targets,
//                                   input_size, state_size, output_size);

        MATH21_PRINT_TIME_ELAPSED(auto dy = grad(params, y));

        m21log("data size", ad_get_data().size());
        if (config.debug && config.functionParasPath_init.empty()) {
            y.log("y", 15);
#if !defined(MATH21_USE_NUMR32)
            VecR y_actual(1);
            y_actual = -4.84005713341334;
            MATH21_PASS(math21_op_isEqual(ad_get_value(y), y_actual, MATH21_EPS));
#endif
        }

        // train
        TenR x_value_cur;
        x_value_cur.setDeviceType(config.deviceType);
        x_value_cur = ad_get_value(params);
        lstm_f_pair_args grad_args;
        grad_args.x = params;
        grad_args.y = dy;
        lstm_f_callback_args callback_args;
        callback_args.printText = config.printTextInOpt;
        callback_args.x = params;
        callback_args.y = y;
        callback_args.logprobs = logprobs;
        callback_args.data_targets = data_targets;

        math21_opt_adam(x_value_cur, df_1d, &grad_args, callback, &callback_args, config.num_iters,
                        config.step_size);

        if (!config.functionParasPath_save.empty()) {
            math21_io_save(config.functionParasPath_save.c_str(), ad_get_value(params));
        }
    }
}