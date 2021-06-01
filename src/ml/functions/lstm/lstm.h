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
#include "../fully_connected/files_c.h"
#include "../dropout/files_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// lstm
// h(t) = lstm(x(t), h(t-1)), t = 1, ..., T.
// h(t), c(t) = f_lstm(x(t), h(t-1), c(t-1)), t = 1, ..., T.
typedef struct mlfunction_lstm mlfunction_lstm;
struct mlfunction_lstm {
    const char* name;
    int implementationMode; // 1, 2, 3
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int batch; // rnn_batch_size, mini_batch_size

    // for input x
    mlfunction_fully_connected *fcWi;
    mlfunction_fully_connected *fcWf;
    mlfunction_fully_connected *fcWo;
    mlfunction_fully_connected *fcWg;

    mlfunction_fully_connected *fcWx; // implementationMode = 2

    // for hidden h
    mlfunction_fully_connected *fcUi;
    mlfunction_fully_connected *fcUf;
    mlfunction_fully_connected *fcUo;
    mlfunction_fully_connected *fcUg;

    mlfunction_fully_connected *fcUh; // implementationMode = 2

    mlfunction_fully_connected *fcW; // implementationMode = 3
    PointerFloatWrapper xh_interleaved; // implementationMode = 3
    PointerFloatWrapper dxh_interleaved; // implementationMode = 3

    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y, here is h(t), t = 1, ..., T.
    PointerFloatWrapper last_output; // Y, here is h(T)

    PointerFloatWrapper h_0;
    PointerFloatWrapper c_0;

    // temp
    PointerFloatWrapper temp;
    PointerFloatWrapper dc_t;

    PointerFloatWrapper i;
    PointerFloatWrapper f;
    PointerFloatWrapper o;
    PointerFloatWrapper g;

    PointerFloatWrapper d_i; // implementationMode = 2
    PointerFloatWrapper d_f; // implementationMode = 2
    PointerFloatWrapper d_o; // implementationMode = 2
    PointerFloatWrapper d_g; // implementationMode = 2
    PointerFloatWrapper ifog_interleaved; // implementationMode = 2
    PointerFloatWrapper difog_interleaved; // implementationMode = 2
    PointerFloatWrapper ifog_noninterleaved; // implementationMode = 2
    PointerFloatWrapper difog_noninterleaved; // implementationMode = 2

    PointerFloatWrapper c; // current c
    PointerFloatWrapper dc_tm1_at_t;
    PointerFloatWrapper cell; // here is c(t), t = 1, ..., T.
    PointerFloatWrapper h; // current hidden state h

    int steps; // n_time_step
    int i_time_step; // time in rnn. todo: check relationship with set_mbs

    NumB is_dropout_x;
    NumB is_dropout_h;
    mlfunction_dropout *dropout_x;
    mlfunction_dropout *dropout_h;
    int is_return_sequences;
};

void math21_ml_function_lstm_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options);

mlfunction_lstm *math21_ml_function_lstm_create(
        mlfunction_node *fnode, int batch, int input_size, int output_size, int n_time_step, int is_use_bias,
        int is_batch_normalize, int is_unit_forget_bias, float dropout_rate_x, float dropout_rate_h, int is_adam,
        int is_return_sequences, int implementationMode);

void math21_ml_function_lstm_forward(mlfunction_lstm *f, mlfunction_node *finput, int is_train);

void math21_ml_function_lstm_backward(mlfunction_lstm *f, mlfunction_node *finput, int is_train);

void math21_ml_function_lstm_update(mlfunction_lstm *f, OptUpdate *optUpdate);

void math21_ml_function_lstm_saveState(const mlfunction_lstm *f, FILE *file);

void math21_ml_function_lstm_increase_by_time(mlfunction_lstm *f, int time_steps);

void math21_ml_function_lstm_reset(mlfunction_lstm *f);

void math21_ml_function_lstm_set_mbs(mlfunction_lstm *f, int mini_batch_size);

void math21_ml_function_lstm_log(const mlfunction_lstm *f, const char *varName);

#ifdef __cplusplus
}
#endif
