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
#include "sin01.h"

namespace math21 {

    // y = sin(2 * pi * freq * (t + t0))
    void generate_sin_sample(MatR &X, MatR &Y, MatR &FX, MatR &FY,
                             NumR f, NumR *pt0, NumN batch_size, NumN n_samples, NumN n_predict) {
        NumR s = 100;
        X.setSize(batch_size, n_samples);
        Y.setSize(batch_size, n_samples);
        FX.setSize(batch_size, n_predict);
        FY.setSize(batch_size, n_predict);

        NumR t0;
        VecR t(n_samples + n_predict);
        t.letters(0);
        math21_op_vector_kx_onto(1 / s, t);
        RanUniform ran;
        VecR y;
        for (NumN i = 1; i <= batch_size; ++i) {
            if (!pt0) {
                math21_random_draw(t0, ran);
                t0 = t0 * 2 * MATH21_PI;
            } else {
                t0 = *pt0 + (i - 1) / (NumR) batch_size;
            }
            if (f == 0) {
                math21_random_draw(f, ran);
                f = f * 3.5 + 0.5;
            }
            math21_op_add(t0, t, y);
            math21_op_vector_kx_onto(2 * MATH21_PI * f, y);
            math21_op_sin_onto(y);
            math21_op_matrix_set_row(t, X, i, 0, 0);
            math21_op_matrix_set_row(t, FX, i, n_samples, 0);
            math21_op_matrix_set_row(y, Y, i, 0, 0);
            math21_op_matrix_set_row(y, FY, i, n_samples, 0);
        }
    }

    namespace ad {
        // x shape: (batch_size, n_steps, n_input)
        ad_point ad_rnn_sin01_predict(ad_point x, const ad_point &theta_rnn, const ad_point &w, ad_point b,
                                      NumN n_input, NumN n_steps, NumN n_hidden, NumN n_outputs) {
            x = ad_axis_swap(x, 1, 2);
            Seqce<ad_point> outputs;
            ad_lstm_only_hiddens(theta_rnn, x, outputs, n_input, n_hidden, n_outputs);
            // linear activation, using rnn inner loop last output
            auto output = outputs(outputs.size());

//            VecR d_value(2);
//            d_value = 1, ad_get_dim_i(b, 1);
//            ad_point d(d_value);
//            b = ad_share_reshape(b, d);

            return ad_add(ad_mat_mul(output, w), b);
        }
    }

    using namespace ad;

    m21lstm_sin01_type_config::m21lstm_sin01_type_config() {
        predict = 0;

//        functionParasPath_init = "sin_paras_init";
        functionParasPath_save = "sin_paras";

        //// Parameters
        display_step = 100;

//        batch_size = 50;
        batch_size = 5;

        input_size = 1;
        state_size = 100;
        output_size = 50;
//        time_steps = 100;
        time_steps = 10;
        deviceType = m21_device_type_default;

        num_iters = 1000;
        step_size = 0.001;

        printTextInOpt = 0;

        n_lines = 20;
        sequence_length = 30;

        debug = 0;
    }

    void m21lstm_sin01_type_config::log(const char *name) const {
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

    typedef struct {
        NumB printText;
        ad_point x;
        ad_point y;
        ad_point data_inputs;
        ad_point data_targets;
        m21lstm_sin01_type_config config;
    } lstm_f_callback_args2;

    static void f_1d(const TenR &x, TenR &y, NumN iter, void *data) {
        auto args = (lstm_f_callback_args2 *) data;
        ad_point px = args->x;
        ad_point py = args->y;
        ad_point data_inputs = args->data_inputs;
        ad_point data_targets = args->data_targets;
        auto config = args->config;

        // prepare data
        MatR T, FT, Y, FY, Y2;
        generate_sin_sample(T, Y, FT, FY, 1, 0, config.batch_size, config.time_steps, config.output_size);
        VecN dx(3);
        dx = config.batch_size, config.time_steps, config.input_size;
        math21_operator_share_reshape(Y, Y2, dx);

        ad_get_value(data_inputs) = Y2;
        ad_get_value(data_targets) = FY;
        ad_get_value(px) = x;
        ad_fv(py);
        y = ad_get_value(py);

        if (iter % 100 == 0) {
            if (!config.functionParasPath_save.empty()) {
                auto s = math21_string_to_string(config.functionParasPath_save, ".", iter);
                math21_io_save(s.c_str(), ad_get_value(px));
            }
        }
    }

    static void df_1d(const TenR &x, TenR &dy, NumN iter, void *data) {
        auto args = (lstm_f_pair_args *) data;
        ad_point px = args->x;
        ad_point pdy = args->y;
        ad_get_value(px) = x;
        ad_fv(pdy);
        dy = ad_get_value(pdy);
    }

    static void callback(const TenR &x_cur, const TenR &gradient, NumN iter, void *data) {
        if (iter % 10 == 1) {
            TenR y;
            f_1d(x_cur, y, iter, data);
            printf("Iteration %d Train loss: %.16lf\n", iter - 1, y(1)); // use below
            std::flush(std::cout);
//        printf("Iteration %d Train loss: %lf\n", iter, y(1));
//        math21_data_text_lstm_print_training_prediction(x_cur, data);
        }
    }

    void math21_ml_lstm_sin01(const m21lstm_sin01_type_config &config) {
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

        // prepare data
        MatR T, FT, Y, FY, Y2;
        generate_sin_sample(T, Y, FT, FY, 1, 0, config.batch_size, config.time_steps, config.output_size);
        VecN dx(3);
        dx = config.batch_size, config.time_steps, config.input_size;
        math21_operator_share_reshape(Y, Y2, dx);

        ad_point data_inputs(Y2, 0, config.deviceType);
        ad_point data_targets(FY, 0, config.deviceType);

        // theta_rnn, w, b
        NumN n_paras;
        NumN n_theta_rnn = ad_lstm_calculate_params_size(config.input_size, config.state_size, config.output_size);
        NumN n_theta_w = config.state_size * config.output_size;
        NumN n_theta_b = config.output_size;
        n_paras = n_theta_rnn + n_theta_w + n_theta_b;

        VecR params_value(n_paras);
        ad_point params(params_value, 1, ad_get_device_type(data_inputs));

        NumN offset = 0;
        auto theta_rnn = ad_vec_share_sub_offset(params, offset, n_theta_rnn);
        offset += n_theta_rnn;
        auto w = ad_vec_share_sub_offset(params, offset, n_theta_w);
        offset += n_theta_w;
        auto b = ad_vec_share_sub_offset(params, offset, n_theta_b);
        VecN d(2);
        d = config.state_size, config.output_size;
        w = ad_share_reshape(w, d);
        d = 1, config.output_size;
        b = ad_share_reshape(b, d);

        // init params
//      ad_rnn_init_params(params, config.input_size, config.state_size, config.output_size, param_scale);
        ad_lstm_init_params(theta_rnn, config.input_size, config.state_size, config.output_size, param_scale);
        RanNormal ranNormal;
        math21_op_random_draw(ad_get_value(w), ranNormal);
        math21_op_random_draw(ad_get_value(b), ranNormal);

        if (!config.functionParasPath_init.empty()) {
            math21_io_load(config.functionParasPath_init.c_str(), ad_get_value(params));
        }

        auto pred = ad_rnn_sin01_predict(data_inputs, theta_rnn, w, b, config.input_size, config.time_steps,
                                         config.state_size,
                                         config.output_size);

        auto t = pred - data_targets;
        t = ad_power(t, 1, 2);
        auto individual_losses = ad_sum(t, 2);
        auto loss = ad_mean(individual_losses);
        const auto &y = loss;

        timer time;
        time.start();

        auto dy = grad(params, y);
        time.end();
        if (time.time() > 0) {
            m21log("\ngrad time used", time.time());
        }

        m21log("data size", ad_get_data().size());

        if (config.debug && config.functionParasPath_init == "") {
            y.log("y", 15);
//if !defined(MATH21_USE_NUMR32)
            VecR y_actual(1);
            y_actual = -4.84005713341334;
            MATH21_PASS(math21_op_isEqual(ad_get_value(y), y_actual, MATH21_EPS));
//endif
        }

        // train
        TenR x_value_cur;
        x_value_cur.setDeviceType(config.deviceType);
        x_value_cur = ad_get_value(params);
        lstm_f_pair_args grad_args;
        grad_args.x = params;
        grad_args.y = dy;
        lstm_f_callback_args2 callback_args;
        callback_args.printText = config.printTextInOpt;
        callback_args.x = params;
        callback_args.y = y;
        callback_args.data_inputs = data_inputs;
        callback_args.data_targets = data_targets;
        callback_args.config = config;

        math21_opt_adam(x_value_cur, df_1d, &grad_args, callback, &callback_args, config.num_iters,
                        config.step_size);

        if (!config.functionParasPath_save.empty()) {
            math21_io_save(config.functionParasPath_save.c_str(), ad_get_value(params));
        }
    }
}