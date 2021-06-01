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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlfunction_rnn mlfunction_rnn;
struct mlfunction_rnn { // .
    const char* name;
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int batch; // mini_batch_size

    int steps;

    mlfunction_fully_connected *input_layer;
    mlfunction_fully_connected *self_layer;
    mlfunction_fully_connected *output_layer;

    PointerFloatWrapper state;
    PointerFloatWrapper prev_state; // only used by backward
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y
};

void math21_ml_function_rnn_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options);

mlfunction_rnn *math21_ml_function_rnn_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        int n_time_step, MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam);

void math21_ml_function_rnn_forward(mlfunction_rnn *f, mlfunction_node *finput, int is_train);

void math21_ml_function_rnn_backward(mlfunction_rnn *f, mlfunction_node *finput, int is_train);

void math21_ml_function_rnn_update(mlfunction_rnn *f, OptUpdate *optUpdate);

void math21_ml_function_rnn_saveState(const mlfunction_rnn *f, FILE *file);

void math21_ml_function_rnn_reset(mlfunction_rnn *f);

void math21_ml_function_rnn_set_mbs(mlfunction_rnn *f, int mini_batch_size);

#ifdef __cplusplus
}
#endif
