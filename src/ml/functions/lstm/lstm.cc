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

#include "inner_cc.h"
#include "lstm.h"

using namespace math21;

void math21_ml_function_lstm_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);
    int is_unit_forget_bias = math21_function_option_find_int_quiet(options, "is_unit_forget_bias", 1);
    float dropout_rate_x = math21_function_option_find_float_quiet(options, "dropout_rate_x", 0);
    float dropout_rate_h = math21_function_option_find_float_quiet(options, "dropout_rate_h", 0);
    int is_return_sequences = math21_function_option_find_int_quiet(options, "is_return_sequences", 0);
    int implementationMode = math21_function_option_find_int_quiet(options, "implementationMode", 1);
    math21_ml_function_lstm_create(fnode, finput->mini_batch_size, finput->y_size, output, fnet->n_time_step_in_rnn,
                                   is_use_bias,
                                   is_batch_normalize, is_unit_forget_bias, dropout_rate_x, dropout_rate_h,
                                   fnet->adam, is_return_sequences, implementationMode);
}

void math21_ml_function_lstm_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_saveState(f, file);
}

// todo: may deprecate this.
void math21_ml_function_lstm_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_set_mbs(f, mini_batch_size);
}

void math21_ml_function_lstm_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_forward(f, finput, net->is_train);
}

void math21_ml_function_lstm_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_backward(f, finput, net->is_train);
}

void math21_ml_function_lstm_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto *f = (mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_update(f, optUpdate);
}

void math21_ml_function_lstm_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_lstm *) fnode->function;
    math21_ml_function_lstm_log(f, varName);
}

const char *math21_ml_function_lstm_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_lstm *) fnode->function;
    return f->name;
}

void math21_ml_function_lstm_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_lstm *) fnode->function;
    fnode->x_dim[0] = f->inputs;
    fnode->x_dim[1] = 1;
    fnode->x_dim[2] = 1;
    fnode->y_dim[0] = f->outputs;
    fnode->y_dim[1] = 1;
    fnode->y_dim[2] = 1;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    if (f->is_return_sequences) {
        fnode->mini_batch_size = f->steps * f->batch;
        fnode->y = f->output;
        fnode->dy = f->delta;
    } else {
        fnode->mini_batch_size = f->batch;
        fnode->y = f->last_output;
        fnode->dy = f->delta + (f->steps - 1) * f->batch * f->outputs;
    }
}

void math21_ml_function_lstm_log(const mlfunction_lstm *f, const char *varName) {
    std::string _varNameNew;
    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
        return;
    }
    varName = _varNameNew.c_str();

    if (math21_string_is_equal(varName, "summary")) {
        varName = "*/summary";
        fprintf(stdout, "lstm: (%d, %d, %d) -> (%d, %d, %d)\n", f->inputs, f->steps, f->batch, f->outputs,
                f->is_return_sequences ? f->steps : 1,
                f->batch);
        if (f->is_dropout_x) {
            fprintf(stdout, "\t\t");
            math21_ml_function_dropout_log(f->dropout_x, varName);
        }
        if (f->is_dropout_h) {
            fprintf(stdout, "\t\t");
            math21_ml_function_dropout_log(f->dropout_h, varName);
        }
        if (f->implementationMode == 1) {
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcWi, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcWf, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcWo, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcWg, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcUi, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcUf, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcUo, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcUg, varName);
        } else if (f->implementationMode == 2) {
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcWx, varName);
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcUh, varName);
        } else {
            fprintf(stdout, "\t\t");
            math21_ml_function_fully_connected_log(f->fcW, varName);
        }
        return;
    }
    if (f->is_dropout_x) {
        math21_ml_function_dropout_log(f->dropout_x, varName);
    }
    if (f->is_dropout_h) {
        math21_ml_function_dropout_log(f->dropout_h, varName);
    }
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_log(f->fcWi, varName);
        math21_ml_function_fully_connected_log(f->fcWf, varName);
        math21_ml_function_fully_connected_log(f->fcWo, varName);
        math21_ml_function_fully_connected_log(f->fcWg, varName);
        math21_ml_function_fully_connected_log(f->fcUi, varName);
        math21_ml_function_fully_connected_log(f->fcUf, varName);
        math21_ml_function_fully_connected_log(f->fcUo, varName);
        math21_ml_function_fully_connected_log(f->fcUg, varName);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_log(f->fcWx, varName);
        math21_ml_function_fully_connected_log(f->fcUh, varName);
    } else {
        math21_ml_function_fully_connected_log(f->fcW, varName);
    }
}

/*
 * many-to-many fashion using a time distributed node with fnode->mini_batch_size = f->steps * f->batch.
 *
 * have done:
 * change order of f and i in bp
 *
 * todo: unit_forget_bias, add implementation mode 2, REMOVE c_tm1 malloc, REMOVE prev_STATE malloc
 * add stateful as in keras.
 *
 * remove c malloc, remove h malloc
 * RENAME prev_STATE TO H_TM1, RENAME PREV_CELL TO C_HM1
 * math21_vector_clip_wrapper IN math21_ml_function_fully_connected_backward
 * implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
 * */
mlfunction_lstm *math21_ml_function_lstm_create(
        mlfunction_node *fnode, int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
        int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
        int is_return_sequences, int implementationMode) {
    int rnn_batch_size = batch / n_time_step;
    auto *f = (mlfunction_lstm *) math21_vector_calloc_cpu(1, sizeof(mlfunction_lstm));

    f->batch = rnn_batch_size;
    f->steps = n_time_step;
    f->inputs = input_size;
    f->outputs = output_size;
    f->is_return_sequences = is_return_sequences;

    f->implementationMode = implementationMode;
//    f->implementationMode = 1;
//    f->implementationMode = 2;
//    f->implementationMode = 3;

    if (dropout_rate_x > 0) {
        f->is_dropout_x = 1;
        mlfunction_node finput_x = {0};
        finput_x.mini_batch_size = n_time_step * rnn_batch_size;
        finput_x.y_size = input_size;
        f->dropout_x = math21_ml_function_dropout_create(0, &finput_x, dropout_rate_x, n_time_step,
                                                         "dropout_x");
    }
    if (dropout_rate_h > 0) {
        f->is_dropout_h = 1;
        mlfunction_node finput_h = {0};
        finput_h.mini_batch_size = n_time_step * rnn_batch_size;
        finput_h.y_size = output_size;
        f->dropout_h = math21_ml_function_dropout_create(0, &finput_h, dropout_rate_h, n_time_step,
                                                         "dropout_h");
    }

    if (f->implementationMode == 1) {
        f->fcWi = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step, input_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             is_use_bias,
                                                                             is_batch_normalize, is_adam, "fcWi");

        f->fcWf = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step, input_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             is_use_bias,
                                                                             is_batch_normalize, is_adam, "fcWf");
        if (f->fcWf->is_use_bias && is_unit_forget_bias) {
            math21_vector_set_wrapper(f->fcWf->outputs, 1, f->fcWf->biases, 1);
        }

        f->fcWo = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step, input_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             is_use_bias,
                                                                             is_batch_normalize, is_adam, "fcWo");

        f->fcWg = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step, input_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             is_use_bias,
                                                                             is_batch_normalize, is_adam, "fcWg");

        f->fcUi = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                             output_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             0,
                                                                             is_batch_normalize, is_adam, "fcUi");

        f->fcUf = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                             output_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             0,
                                                                             is_batch_normalize, is_adam, "fcUf");

        f->fcUo = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                             output_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             0,
                                                                             is_batch_normalize, is_adam, "fcUo");

        f->fcUg = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                             output_size,
                                                                             output_size,
                                                                             MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                             0,
                                                                             is_batch_normalize, is_adam, "fcUg");

        f->i = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->f = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->o = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
        f->g = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    } else {
        if (f->implementationMode == 2) {
            f->fcWx = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                                 input_size,
                                                                                 output_size * 4,
                                                                                 MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                                 is_use_bias,
                                                                                 is_batch_normalize, is_adam, "fcWx");
            if (f->fcWx->is_use_bias && is_unit_forget_bias) {
                math21_vector_set_wrapper(f->fcWx->outputs / 4, 1, f->fcWx->biases + f->fcWx->outputs / 4, 1);
            }

            f->fcUh = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                                 output_size,
                                                                                 output_size * 4,
                                                                                 MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                                 0,
                                                                                 is_batch_normalize, is_adam, "fcUh");
        } else {
            MATH21_ASSERT(f->implementationMode == 3);
            f->fcW = math21_ml_function_fully_connected_with_n_time_step_create(0, rnn_batch_size, n_time_step,
                                                                                input_size + output_size,
                                                                                output_size * 4,
                                                                                MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR,
                                                                                is_use_bias,
                                                                                is_batch_normalize, is_adam, "fcW");
            if (f->fcW->is_use_bias && is_unit_forget_bias) {
                math21_vector_set_wrapper(f->fcW->outputs / 4, 1, f->fcW->biases + f->fcW->outputs / 4, 1);
            }
            f->xh_interleaved = math21_vector_create_with_default_value_wrapper(
                    rnn_batch_size * (input_size + output_size), 0);
            f->dxh_interleaved = math21_vector_create_with_default_value_wrapper(
                    rnn_batch_size * (input_size + output_size), 0);
        }

        // shape: (mbs, 4, y_size)
        f->ifog_interleaved = math21_vector_create_with_default_value_wrapper(4 * rnn_batch_size * output_size, 0);
        // shape: (4, mbs, y_size)
        f->ifog_noninterleaved = math21_vector_create_with_default_value_wrapper(4 * rnn_batch_size * output_size, 0);
        f->i = f->ifog_noninterleaved;
        f->f = f->i + rnn_batch_size * output_size;
        f->o = f->f + rnn_batch_size * output_size;
        f->g = f->o + rnn_batch_size * output_size;

        f->difog_interleaved = f->ifog_noninterleaved;
        f->difog_noninterleaved = f->ifog_interleaved;
        f->d_i = f->difog_noninterleaved;
        f->d_f = f->d_i + rnn_batch_size * output_size;
        f->d_o = f->d_f + rnn_batch_size * output_size;
        f->d_g = f->d_o + rnn_batch_size * output_size;
    }

    f->output = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);
    f->last_output = f->output + (n_time_step - 1) * rnn_batch_size * output_size;
    f->delta = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);
    f->cell = math21_vector_create_with_default_value_wrapper(n_time_step * rnn_batch_size * output_size, 0);

    f->h_0 = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    f->c_0 = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

    f->c = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    // todo: maybe consider h to be pointer of output, but must solve mlfnode_type_rnn first.
    f->h = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    f->temp = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    f->dc_t = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);
    f->dc_tm1_at_t = math21_vector_create_with_default_value_wrapper(rnn_batch_size * output_size, 0);

    f->name = math21_string_create_from_string("lstm");
    if (fnode) {
        fnode->type = mlfnode_type_lstm;
        fnode->function = f;
        fnode->saveState = math21_ml_function_lstm_node_saveState;
        fnode->set_mbs = math21_ml_function_lstm_node_set_mbs;
        fnode->forward = math21_ml_function_lstm_node_forward;
        fnode->backward = math21_ml_function_lstm_node_backward;
        fnode->update = math21_ml_function_lstm_node_update;
        fnode->log = math21_ml_function_lstm_node_log;
        fnode->getName = math21_ml_function_lstm_node_getName;
        math21_ml_function_lstm_node_reset(fnode);
    }
    return f;
}

void _math21_ml_function_lstm_ifog_transpose(mlfunction_lstm *f, NumB isForward) {
    if (isForward) {
//        math21_tensor_3d_float_log_wrapper("f->ifog_interleaved", f->ifog_interleaved, f->batch, 4, f->outputs);
        math21_vector_transpose_d1234_to_d1324_wrapper(f->ifog_interleaved, f->ifog_noninterleaved,
                                                       1, f->batch, 4, f->outputs);
//        math21_tensor_3d_float_log_wrapper("f->ifog_noninterleaved", f->ifog_noninterleaved, 4, f->batch, f->outputs);
    } else {
        math21_vector_transpose_d1234_to_d1324_wrapper(f->difog_noninterleaved, f->difog_interleaved,
                                                       1, 4, f->batch, f->outputs);
    }
}

// not rigorous tensor transpose
void _math21_ml_function_lstm_xh_set(mlfunction_lstm *f,
                                     PointerFloatWrapper x, PointerFloatWrapper h,
                                     PointerFloatWrapper xh, NumB isForward) {
    if (isForward) {
        if (!math21_vector_isEmpty_wrapper(x)){
            math21_vector_assign_3d_d2_wrapper(x, xh, f->batch, f->inputs + f->outputs, 1, f->inputs, 0, 0);
        }
        if (!math21_vector_isEmpty_wrapper(h)) {
            math21_vector_assign_3d_d2_wrapper(h, xh, f->batch, f->inputs + f->outputs, 1, f->outputs, f->inputs, 0);
        }
    } else {
        if (!math21_vector_isEmpty_wrapper(x)) {
            math21_vector_assign_3d_d2_wrapper(xh, x, f->batch, f->inputs + f->outputs, 1, f->inputs, 0, 1);
        }
        if (!math21_vector_isEmpty_wrapper(h)) {
            math21_vector_assign_3d_d2_wrapper(xh, h, f->batch, f->inputs + f->outputs, 1, f->outputs, f->inputs, 1);
        }
    }
}

void _math21_ml_function_lstm_x_add_h_and_activate(mlfunction_lstm *f, NumB isForward) {
    if (f->implementationMode == 1) {
        // y_i = y_h + y_x for input gate
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWi->output, 1, f->i, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUi->output, 1, f->i, 1);
        // y_f = y_h + y_x for forget gate
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWf->output, 1, f->f, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUf->output, 1, f->f, 1);
        // y_o = y_h + y_x for output gate
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWo->output, 1, f->o, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUo->output, 1, f->o, 1);
        // y_g = y_h + y_x for cell
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->fcWg->output, 1, f->g, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->fcUg->output, 1, f->g, 1);
    } else if (f->implementationMode == 2) {
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch * 4, f->fcWx->output, 1, f->ifog_interleaved, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch * 4, 1, f->fcUh->output, 1, f->ifog_interleaved, 1);
        _math21_ml_function_lstm_ifog_transpose(f, 1);
    } else {
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch * 4, f->fcW->output, 1, f->ifog_interleaved, 1);
        if(math21_ml_function_tool_is_debug() && isForward){
//            math21_tensor_2d_float_log_wrapper("ifog_interleaved", f->ifog_interleaved, f->batch, f->outputs * 4);
        }
        _math21_ml_function_lstm_ifog_transpose(f, 1);
    }
    // i = y_i <- sigmoid(y_i)
    math21_function_activation_vector_wrapper(f->i, f->outputs * f->batch,
                                              MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
    // f = y_f <- sigmoid(y_f)
    math21_function_activation_vector_wrapper(f->f, f->outputs * f->batch,
                                              MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
    // o = y_o <- sigmoid(y_o)
    math21_function_activation_vector_wrapper(f->o, f->outputs * f->batch,
                                              MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC);
    // g = y_g <- tanh(y_g)
    math21_function_activation_vector_wrapper(f->g, f->outputs * f->batch, MATH21_FUNCTION_ACTIVATION_TYPE_TANH);
}

/*
h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.

(i, f, o, g)' = (sigm, sigm, sigm, tanh)' (W *(x(t), h(t-1))')
// CEC
c(t) = f * c(t-1) + i * g
h(t) = o * tanh(c(t))
with W :=
            Wi, Ui,
            Wf, Uf,
            Wo, Uo,
            Wg, Ug;

dropout:
(i, f, o, g)' = (sigm, sigm, sigm, tanh)' (W *(x(t)*zx, h(t-1)*zh)')
with zx, zh random masks repeated at all time steps. (Note: zx, zh is independent of t)

// the following is not used
y(t) = Why*h(t) + by
y = W_hy * h(t) + b_y
h_pre, h(1), ..., h(t), ..., h(T)

# References
        - [Long short-term memory](
          http://www.bioinf.jku.at/publications/older/2604.pdf)
        - [Learning to forget: Continual prediction with LSTM](
          http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [A Theoretically Grounded Application of Dropout in
          Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
 * */
void math21_ml_function_lstm_forward(mlfunction_lstm *f, mlfunction_node *finput, int is_train) {
    int i;

    if (is_train) {
        math21_vector_set_wrapper(f->steps * f->batch * f->outputs, 0, f->delta, 1);
        math21_vector_set_wrapper(f->batch * f->outputs, 0, f->dc_tm1_at_t, 1);
        // h_0 <- h, c_0 <- c, backup for backward
        math21_vector_assign_from_vector_wrapper(f->batch * f->outputs, f->h, 1, f->h_0, 1);
        math21_vector_assign_from_vector_wrapper(f->batch * f->outputs, f->c, 1, f->c_0, 1);
    }

    mlfunction_node finput_fc_x0 = {0};
    mlfunction_node *finput_fc_x = &finput_fc_x0;
    mlfunction_node finput_fc_h0 = {0};
    mlfunction_node *finput_fc_h = &finput_fc_h0;
    mlfunction_node finput_fc_xh0 = {0};
    mlfunction_node *finput_fc_xh = &finput_fc_xh0;
    for (i = 0; i < f->steps; ++i) {
        if (f->is_dropout_x) {
            finput_fc_x->y = finput->y + i * f->inputs * f->batch;
            math21_ml_function_dropout_forward(f->dropout_x, finput_fc_x, is_train);
            finput_fc_x->y = f->dropout_x->y;
        } else {
            finput_fc_x->y = finput->y + i * f->inputs * f->batch;
        }

        if (f->is_dropout_h) {
            finput_fc_h->y = f->h;
            math21_ml_function_dropout_forward(f->dropout_h, finput_fc_h, is_train);
            finput_fc_h->y = f->dropout_h->y;
        } else {
            finput_fc_h->y = f->h;
        }

        if (f->implementationMode == 1) {
            math21_ml_function_fully_connected_forward(f->fcWi, finput_fc_x, is_train);
            math21_ml_function_fully_connected_forward(f->fcWf, finput_fc_x, is_train);
            math21_ml_function_fully_connected_forward(f->fcWo, finput_fc_x, is_train);
            math21_ml_function_fully_connected_forward(f->fcWg, finput_fc_x, is_train);

            math21_ml_function_fully_connected_forward(f->fcUi, finput_fc_h, is_train);
            math21_ml_function_fully_connected_forward(f->fcUf, finput_fc_h, is_train);
            math21_ml_function_fully_connected_forward(f->fcUo, finput_fc_h, is_train);
            math21_ml_function_fully_connected_forward(f->fcUg, finput_fc_h, is_train);
        } else if (f->implementationMode == 2) {
            math21_ml_function_fully_connected_forward(f->fcWx, finput_fc_x, is_train);
            math21_ml_function_fully_connected_forward(f->fcUh, finput_fc_h, is_train);
        } else {
            // finput_fc_x, finput_fc_h -> finput_fc_xh
            _math21_ml_function_lstm_xh_set(f, finput_fc_x->y, finput_fc_h->y,
                                            f->xh_interleaved, 1);
            finput_fc_xh->y = f->xh_interleaved;

            math21_ml_function_fully_connected_forward(f->fcW, finput_fc_xh, is_train);
        }
        _math21_ml_function_lstm_x_add_h_and_activate(f, 1);

        // CEC: c(t) = f * c(t-1) + i * g
        // i * g
        // c <- f * c = f * c(t-1)
        // c <- c + i * g
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->i, 1, f->temp, 1);
        math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, f->temp, 1);
        math21_vector_xy_wrapper(f->outputs * f->batch, f->f, 1, f->c, 1);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->temp, 1, f->c, 1);
        if(math21_ml_function_tool_is_debug()){
//            math21_tensor_2d_float_log_wrapper("c(t)", f->c, f->batch, f->outputs);
        }

        // h(t) = o * tanh(c(t))
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->c, 1, f->h, 1);
        math21_function_activation_vector_wrapper(f->h, f->outputs * f->batch, MATH21_FUNCTION_ACTIVATION_TYPE_TANH);
        math21_vector_xy_wrapper(f->outputs * f->batch, f->o, 1, f->h, 1);
        if(math21_ml_function_tool_is_debug()){
//            math21_tensor_2d_float_log_wrapper("h(t)", f->h, f->batch, f->outputs);
        }


        // cell(t) <- c = c(t)
        // y = h = h(t)
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->c, 1, f->cell, 1);
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->h, 1, f->output, 1);

        math21_ml_function_lstm_increase_by_time(f, 1);
    }
    math21_ml_function_lstm_reset(f);
}

// h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.
// dy => dW, dx
// BPTT
// truncated BPTT
// here not using RTRL
void math21_ml_function_lstm_backward(mlfunction_lstm *f, mlfunction_node *finput0, int is_train) {
    int i;

    math21_ml_function_lstm_increase_by_time(f, f->steps - 1);

    mlfunction_node finput_fc_x0 = {0};
    mlfunction_node *finput_fc_x = &finput_fc_x0;
    mlfunction_node finput_fc_h0 = {0};
    mlfunction_node *finput_fc_h = &finput_fc_h0;
    mlfunction_node finput_fc_xh0 = {0};
    mlfunction_node *finput_fc_xh = &finput_fc_xh0;
    mlfunction_node finput1 = *finput0;
    mlfunction_node *finput = &finput1;

    finput->y += f->inputs * f->batch * (f->steps - 1);
    if (!math21_vector_isEmpty_wrapper(finput->dy)) finput->dy += f->inputs * f->batch * (f->steps - 1);

    for (i = f->steps - 1; i >= 0; --i) {
        // c_tm1 = cell(t-1)
        PointerFloatWrapper c_tm1 = (i == 0) ? f->c_0 : f->cell - f->outputs * f->batch;
        // c_t = c = cell(t)
        PointerFloatWrapper c_t = f->cell;

        // h_tm1 = h(t-1) = y(t-1)
        PointerFloatWrapper h_tm1 = (i == 0) ? f->h_0 : f->output - f->outputs * f->batch;
        // dL/dh(t-1)
        // dh_tm1 = dh(t-1) = dy(t-1)
        // truncated at t = 1, i.e., i=0
        PointerFloatWrapper dh_tm1 = (i == 0) ? math21_vector_getEmpty_R32_wrapper() : f->delta - f->outputs * f->batch;
        // get ifog at t
        _math21_ml_function_lstm_x_add_h_and_activate(f, 0);

        // CEC
        // c(t) = f * c(t-1) + i * g
        // h(t) = y(t) = o * tanh(c(t))

        //// tanh(c(t))
        // temp = tanh(c(t))
        auto &tanh_c_t = f->temp;
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, c_t, 1, tanh_c_t, 1);
        math21_function_activation_vector_wrapper(tanh_c_t, f->outputs * f->batch,
                                                  MATH21_FUNCTION_ACTIVATION_TYPE_TANH);

        //// dc(t)
        // h(t) = y(t) = o * tanh(c(t))
        // o * dy(t)
        // dc(t) = d(tanh) * o * dy(t)
        // dc(t) <- dc_tm1_at_t + dc(t), here dc_tm1_at_t = dc(t) at t + 1
        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->delta, 1, f->dc_t, 1);
        math21_vector_xy_wrapper(f->outputs * f->batch, f->o, 1, f->dc_t, 1);
        math21_function_activation_gradient_vector_wrapper(tanh_c_t, f->outputs * f->batch,
                                                           MATH21_FUNCTION_ACTIVATION_TYPE_TANH, f->dc_t);
        math21_vector_kx_add_y_wrapper(f->outputs * f->batch, 1, f->dc_tm1_at_t, 1, f->dc_t, 1);

        //// c
        // c(t) = f * c(t-1) + i * g
        // dc(t)
        // dc(t-1) at t = f * dc(t)
        // dc_tm1_at_t = dc(t-1) at t
        if (i > 0) {
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f->dc_tm1_at_t, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->f, 1, f->dc_tm1_at_t, 1);
        }

        if (f->implementationMode == 1) {
            //// o
            // h(t) = y(t) = o * tanh(c(t))
            // temp = tanh(c(t))
            // do = tanh(c(t)) * dy
            // dneto = d(sigm) * do
            auto &o_temp = f->fcWo->delta;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, tanh_c_t, 1, o_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->delta, 1, o_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->o, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, o_temp);
            // dneto
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, o_temp, 1, f->fcWo->delta, 1);
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, o_temp, 1, f->fcUo->delta, 1);

            //// g
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // dg = i * dc(t)
            // dnetg = d(tanh) * dg
            auto &g_temp = f->fcWg->delta;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, g_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->i, 1, g_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->g, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_TANH, g_temp);
            // dnetg
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, g_temp, 1, f->fcWg->delta, 1);
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, g_temp, 1, f->fcUg->delta, 1);

            //// f
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // df = c(t-1) * dc(t)
            // dnetf = d(sigm) * df
            auto &f_temp = f->fcWf->delta;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, c_tm1, 1, f_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->f, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, f_temp);
            // dnetf
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f_temp, 1, f->fcWf->delta, 1);
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f_temp, 1, f->fcUf->delta, 1);

            //// i
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // di = g * dc(t)
            // dneti = d(sigm) * di
            auto &i_temp = f->fcWi->delta;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, i_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, i_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->i, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, i_temp);
            // dneti
//        math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, i_temp, 1, f->fcWi->delta, 1);
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, i_temp, 1, f->fcUi->delta, 1);
        } else {
//// o
            // h(t) = y(t) = o * tanh(c(t))
            // temp = tanh(c(t))
            // do = tanh(c(t)) * dy
            // dneto = d(sigm) * do
            auto &o_temp = f->d_o;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, tanh_c_t, 1, o_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->delta, 1, o_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->o, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, o_temp);

            //// g
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // dg = i * dc(t)
            // dnetg = d(tanh) * dg
            auto &g_temp = f->d_g;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, g_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->i, 1, g_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->g, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_TANH, g_temp);

            //// f
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // df = c(t-1) * dc(t)
            // dnetf = d(sigm) * df
            auto &f_temp = f->d_f;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, f_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, c_tm1, 1, f_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->f, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, f_temp);

            //// i
            // c(t) = f * c(t-1) + i * g
            // dc(t)
            // di = g * dc(t)
            // dneti = d(sigm) * di
            auto &i_temp = f->d_i;
            math21_vector_assign_from_vector_wrapper(f->outputs * f->batch, f->dc_t, 1, i_temp, 1);
            math21_vector_xy_wrapper(f->outputs * f->batch, f->g, 1, i_temp, 1);
            math21_function_activation_gradient_vector_wrapper(f->i, f->outputs * f->batch,
                                                               MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, i_temp);


            _math21_ml_function_lstm_ifog_transpose(f, 0);

            if (f->implementationMode == 2) {
                // dnet
                math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                         f->fcWx->delta,
                                                         1);
                math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                         f->fcUh->delta,
                                                         1);
            } else {
                // dnet
                math21_vector_assign_from_vector_wrapper(4 * f->outputs * f->batch, f->difog_interleaved, 1,
                                                         f->fcW->delta,
                                                         1);
            }
        }

        ////
        if (f->is_dropout_x) {
            finput_fc_x->y = f->dropout_x->y;
            finput_fc_x->dy = f->dropout_x->dy;
        } else {
            finput_fc_x->y = finput->y;
            finput_fc_x->dy = finput->dy;
        }

        if (f->is_dropout_h) {
            finput_fc_h->y = f->dropout_h->y;
            finput_fc_h->dy = f->dropout_h->dy;
        } else {
            finput_fc_h->y = h_tm1;
            finput_fc_h->dy = dh_tm1;
        }

        if (f->implementationMode == 1) {
            // dneto => dwo, duo, dx, dh
            math21_ml_function_fully_connected_backward(f->fcWo, finput_fc_x, is_train);
            math21_ml_function_fully_connected_backward(f->fcUo, finput_fc_h, is_train);
            // dnetg => dwg, dug, dx, dh
            math21_ml_function_fully_connected_backward(f->fcWg, finput_fc_x, is_train);
            math21_ml_function_fully_connected_backward(f->fcUg, finput_fc_h, is_train);
            // dnetf => dwf, duf, dx, dh
            math21_ml_function_fully_connected_backward(f->fcWf, finput_fc_x, is_train);
            math21_ml_function_fully_connected_backward(f->fcUf, finput_fc_h, is_train);
            // dneti => dwi, dui, dx, dh
            math21_ml_function_fully_connected_backward(f->fcWi, finput_fc_x, is_train);
            math21_ml_function_fully_connected_backward(f->fcUi, finput_fc_h, is_train);
        } else if (f->implementationMode == 2) {
            // dnet => dw, du, dx, dh
            math21_ml_function_fully_connected_backward(f->fcWx, finput_fc_x, is_train);
            math21_ml_function_fully_connected_backward(f->fcUh, finput_fc_h, is_train);
        } else {
            _math21_ml_function_lstm_xh_set(f, finput_fc_x->y, finput_fc_h->y,
                                            f->xh_interleaved, 1);
            _math21_ml_function_lstm_xh_set(f, finput_fc_x->dy, finput_fc_h->dy,
                                            f->dxh_interleaved, 1);
            finput_fc_xh->y = f->xh_interleaved;
            finput_fc_xh->dy = f->dxh_interleaved;

            // dnet => dw, dx, dh
            math21_ml_function_fully_connected_backward(f->fcW, finput_fc_xh, is_train);

            _math21_ml_function_lstm_xh_set(f, finput_fc_x->dy, finput_fc_h->dy,
                                            f->dxh_interleaved, 0);
        }

        if (f->is_dropout_x) {
            finput_fc_x->y = finput->y;
            finput_fc_x->dy = finput->dy;
            math21_ml_function_dropout_backward(f->dropout_x, finput_fc_x);
        }
        if (f->is_dropout_h) {
            finput_fc_h->y = h_tm1;
            finput_fc_h->dy = dh_tm1;
            math21_ml_function_dropout_backward(f->dropout_h, finput_fc_h);
        }

        finput->y -= f->inputs * f->batch;
        if (!math21_vector_isEmpty_wrapper(finput->dy)) finput->dy -= f->inputs * f->batch;
        math21_ml_function_lstm_increase_by_time(f, -1);
    }
    math21_ml_function_lstm_reset(f);

//    math21_ml_function_debug_function_save_state(f, 0);
}

void math21_ml_function_lstm_update(mlfunction_lstm *f, OptUpdate *optUpdate) {
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_update(f->fcWi, optUpdate);
        math21_ml_function_fully_connected_update(f->fcWf, optUpdate);
        math21_ml_function_fully_connected_update(f->fcWo, optUpdate);
        math21_ml_function_fully_connected_update(f->fcWg, optUpdate);
        math21_ml_function_fully_connected_update(f->fcUi, optUpdate);
        math21_ml_function_fully_connected_update(f->fcUf, optUpdate);
        math21_ml_function_fully_connected_update(f->fcUo, optUpdate);
        math21_ml_function_fully_connected_update(f->fcUg, optUpdate);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_update(f->fcWx, optUpdate);
        math21_ml_function_fully_connected_update(f->fcUh, optUpdate);
    } else {
        math21_ml_function_fully_connected_update(f->fcW, optUpdate);
    }
}

void math21_ml_function_lstm_saveState(const mlfunction_lstm *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->steps * f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->steps * f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->cell, f->steps * f->batch * f->outputs);
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_saveState(f->fcWi, file);
        math21_ml_function_fully_connected_saveState(f->fcWf, file);
        math21_ml_function_fully_connected_saveState(f->fcWo, file);
        math21_ml_function_fully_connected_saveState(f->fcWg, file);
        math21_ml_function_fully_connected_saveState(f->fcUi, file);
        math21_ml_function_fully_connected_saveState(f->fcUf, file);
        math21_ml_function_fully_connected_saveState(f->fcUo, file);
        math21_ml_function_fully_connected_saveState(f->fcUg, file);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_saveState(f->fcWx, file);
        math21_ml_function_fully_connected_saveState(f->fcUh, file);
    } else {
        math21_ml_function_fully_connected_saveState(f->fcW, file);
    }
}

void math21_ml_function_lstm_increase_by_time(mlfunction_lstm *f, int time_steps) {
    f->i_time_step += time_steps;
    int num = f->outputs * f->batch * time_steps;
    f->output += num;
    f->delta += num;
    f->cell += num;
    if (f->is_dropout_x) {
        math21_ml_function_dropout_increase_by_time(f->dropout_x, time_steps);
    }
    if (f->is_dropout_h) {
        math21_ml_function_dropout_increase_by_time(f->dropout_h, time_steps);
    }
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_increase_by_time(f->fcWi, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcWf, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcWo, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcWg, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcUi, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcUf, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcUo, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcUg, time_steps);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_increase_by_time(f->fcWx, time_steps);
        math21_ml_function_fully_connected_increase_by_time(f->fcUh, time_steps);
    } else {
        math21_ml_function_fully_connected_increase_by_time(f->fcW, time_steps);
    }
}

void math21_ml_function_lstm_reset(mlfunction_lstm *f) {
    int num = f->outputs * f->batch * f->i_time_step;
    f->output -= num;
    f->delta -= num;
    f->cell -= num;
    f->i_time_step = 0;

    if (f->is_dropout_x) {
        math21_ml_function_dropout_reset(f->dropout_x);
    }
    if (f->is_dropout_h) {
        math21_ml_function_dropout_reset(f->dropout_h);
    }
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_reset(f->fcWi);
        math21_ml_function_fully_connected_reset(f->fcWf);
        math21_ml_function_fully_connected_reset(f->fcWo);
        math21_ml_function_fully_connected_reset(f->fcWg);
        math21_ml_function_fully_connected_reset(f->fcUi);
        math21_ml_function_fully_connected_reset(f->fcUf);
        math21_ml_function_fully_connected_reset(f->fcUo);
        math21_ml_function_fully_connected_reset(f->fcUg);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_reset(f->fcWx);
        math21_ml_function_fully_connected_reset(f->fcUh);
    } else {
        math21_ml_function_fully_connected_reset(f->fcW);
    }
}

void math21_ml_function_lstm_set_mbs(mlfunction_lstm *f, int mini_batch_size) {
    f->batch = mini_batch_size;
    if (f->implementationMode == 1) {
        math21_ml_function_fully_connected_set_mbs(f->fcWi, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcWf, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcWo, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcWg, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcUi, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcUf, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcUo, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcUg, mini_batch_size);
    } else if (f->implementationMode == 2) {
        math21_ml_function_fully_connected_set_mbs(f->fcWx, mini_batch_size);
        math21_ml_function_fully_connected_set_mbs(f->fcUh, mini_batch_size);
    } else {
        math21_ml_function_fully_connected_set_mbs(f->fcW, mini_batch_size);
    }
}