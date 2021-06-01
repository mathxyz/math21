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

#ifdef __cplusplus
extern "C" {
#endif

struct mlfunction_net {
    const char *name;
    NumSize n_seen; //
    int mini_batch_size; // mini-batch size in function. It will contain time if there is time. Here time is n_time_step_in_rnn.
    int mbs_y; // mini-batch size in function. It may contain time if there is time. Here time is n_time_step_in_rnn.
    // note: mini_batch_size_in_opt = k * mini_batch_size, k is integer.
    int mini_batch_size_in_opt; // mini-batch size in sgd
    int n_time_step_in_rnn;
    int n_mini_batch_max_in_opt;
    int n_node;
    mlfunction_node **nodes;

    PointerFloatWrapper workspace;// X_prime or dL/dX_prime
    int is_train;
    mlfunction_node *finput; // put temporary.
    mlfunction_node *ftruth; // put temporary, may be changed by fnode->function.

    int data_x_dim[MATH21_DIMS_RAW_TENSOR];

    PointerFloatWrapper data_x_wrapper;
    float *data_x_cpu;

    PointerFloatWrapper data_y_wrapper;
    float *data_y_cpu;

    int data_x_size; // may have value and data_x_dim is not set.
    int data_y_size;

    int y_size;
    PointerFloatWrapper y_wrapper;
    float *y_cpu;

    int gpuDevice;
    float cost;

//    OptUpdate optUpdate;
//    OptUpdate_Adam optUpdateAdam;

    int time_step_in_opt;

    OptAlphaPolicy alphaPolicy;
    int burn_in; // alpha
    float alpha; // learning rate
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;

    // OptAlphaPolicy_STEPS
    int num_steps;
    int *steps;
    float *scales;

    float gamma;
    int step;
    float power;
    float scale;

    // pre-processing
    float exposure;
    float saturation;
    float hue;

    void *detail;
};

// create, parse, resize, forward, backward, update, sync, read, write

mlfunction_net *math21_ml_function_net_create(int n_node);

void math21_ml_function_net_destroy(mlfunction_net *net);

mlfunction_node *math21_ml_function_net_get_output_node(mlfunction_net *net);

void math21_ml_function_net_calculate_cost(mlfunction_net *fnet);

#ifndef MATH21_FLAG_USE_CPU

void math21_ml_function_net_pull_output_wrapper(mlfunction_net *net);

#endif

NumN math21_ml_function_net_getlogLevel();

void math21_ml_function_net_setlogLevel(NumN logLevel);

void math21_ml_function_net_log(const mlfunction_net *fnet);

void math21_ml_function_net_forward(mlfunction_net *net);

void math21_ml_function_net_backward(mlfunction_net *net);

NumSize math21_ml_function_net_get_update_count(mlfunction_net *fnet);

int math21_ml_function_net_should_train_continue(mlfunction_net *fnet);

float math21_ml_function_net_opt_get_alpha(mlfunction_net *fnet);

void math21_ml_function_net_opt_update(mlfunction_net *fnet);

float math21_ml_function_net_train_one_mini_batch_in_function(mlfunction_net *fnet);

float math21_ml_function_net_train_single(mlfunction_net *fnet, m21data d);

float *math21_ml_function_net_predict_input(mlfunction_net *fnet, float *input);

void math21_ml_function_net_log_opt_paras(mlfunction_net *fnet);

void math21_ml_function_net_set_mbs(mlfunction_net *fnet, int mbs);

float *math21_ml_function_net_predict_image(mlfunction_net *fnet, m21image m);

// feed data to net cpu.
void math21_ml_function_net_data_feed(mlfunction_net *fnet, const float *x, const float *y);

void math21_ml_function_net_node_log_by_name(const mlfunction_net *fnet, NumN nodeId, const char *varName);

const void *math21_ml_function_net_node_get_data_to_cpu(mlfunction_net *fnet, NumN nodeId, const char *varName);

m21rawtensor math21_ml_function_net_node_get_rawtensor_to_cpu(mlfunction_net *fnet, NumN nodeId, const char *varName);

#ifdef __cplusplus
}
#endif
