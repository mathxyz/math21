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
#include "../batch_normalization/files_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlfunction_fully_connected mlfunction_fully_connected;
struct mlfunction_fully_connected {
    const char* name;
    float learning_rate_scale;
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int batch; // mini_batch_size
    int h, w, c; // nr_X, nc_X, nch_X
    int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y, shape: n_time_step * mbs * nr_Y * nc_Y * nch_Y
    PointerFloatWrapper weight_updates; // dL/dW
    PointerFloatWrapper bias_updates; // dL/db
    PointerFloatWrapper weights; // W
    float *weights_cpu; // W
    PointerFloatWrapper biases; // b
    float *biases_cpu; // b
    int is_use_bias; // applied only to bias when not using batch normalization.

    PointerFloatWrapper m;
    PointerFloatWrapper v;
    PointerFloatWrapper bias_m;
    PointerFloatWrapper bias_v; // todo: remove when is_use_bias is 0

    int nweights;
    MATH21_FUNCTION_ACTIVATION_TYPE activation;
    mlfunction_batchnorm *bn;
    int flipped; // now just kept for loading dk paras.

    int total_mbs; // n_time_step * mini_batch_size, created in memory
    int n_time_step;
    int i_time_step; // time in rnn. todo: check relationship with set_mbs
};

void math21_ml_function_fully_connected_merge_to(mlfunction_fully_connected *f, mlfunction_fully_connected *fb);
void math21_ml_function_fully_connected_scale(mlfunction_fully_connected *f, float s);
void math21_ml_function_fully_connected_pull_wrapper(mlfunction_fully_connected *l, NumB useRolling);
void math21_ml_function_fully_connected_push_wrapper(mlfunction_fully_connected *l, NumB useRolling);
void math21_ml_function_fully_connected_push_by_wrapper(mlfunction_fully_connected *f, mlfunction_fully_connected *fb, NumB useRolling);

void math21_ml_function_fully_connected_save_theta_order_bwsmv(mlfunction_fully_connected *l, FILE *fp);
void math21_ml_function_fully_connected_load_theta_order_bwsmv(mlfunction_fully_connected *l, FILE *fp);
void math21_ml_function_fully_connected_load_theta_order_bwsmv_flipped(mlfunction_fully_connected *l, FILE *fp, int flipped);

void math21_ml_function_fully_connected_save_theta(mlfunction_fully_connected *l, FILE *fp);
void math21_ml_function_fully_connected_load_theta(mlfunction_fully_connected *l, FILE *fp);

void math21_ml_function_fully_connected_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                              const mlfunction_node *finput, m21list *options);

void math21_ml_function_fully_connected_log(const mlfunction_fully_connected *f, const char *varName);

mlfunction_fully_connected *math21_ml_function_fully_connected_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam, const char* name);

mlfunction_fully_connected *math21_ml_function_fully_connected_with_n_time_step_create(
        mlfunction_node *fnode, int rnn_batch_size, int n_time_step, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam, const char* name);

void math21_ml_function_fully_connected_forward(
        mlfunction_fully_connected *f, mlfunction_node *finput, int is_train);

void math21_ml_function_fully_connected_backward(
        mlfunction_fully_connected *f, mlfunction_node *finput, int is_train);

void math21_ml_function_fully_connected_update(mlfunction_fully_connected *f, OptUpdate *optUpdate);

void math21_ml_function_fully_connected_saveState(const mlfunction_fully_connected *f, FILE *file);

void math21_ml_function_fully_connected_increase_by_time(mlfunction_fully_connected *f, int time_steps);

void math21_ml_function_fully_connected_reset(mlfunction_fully_connected *f);

// todo: may deprecate
void math21_ml_function_fully_connected_set_mbs(mlfunction_fully_connected *f, int mini_batch_size);

#ifdef __cplusplus
}
#endif
