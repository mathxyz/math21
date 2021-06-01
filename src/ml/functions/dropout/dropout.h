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

typedef struct mlfunction_dropout mlfunction_dropout;
struct mlfunction_dropout { // .
    const char *name;
    int batch; // mini_batch_size
    int y_dim[MATH21_DIMS_RAW_TENSOR];
    int outputs; // l.out_h*l.out_w*l.out_c;
    int inputs; // l.out_h*l.out_w*l.out_c;

    PointerFloatWrapper y; // Y
    PointerFloatWrapper dy; // dL/dY

    // With probability `rate`, drops elements of `x`. Input that are kept are
    //  scaled up by `scale = 1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
    //  the expected sum is unchanged.
    float rate; // The probability that each element is dropped. For example, setting rate=0.1 would drop 10% of input elements.
    float scale; // tf.nn.dropout
    PointerFloatWrapper rand; // rand(i) in [0, 1], (Note: it is independent of t when in rnn,
    // and in this case its size is 1/T of usual)

    int total_mbs; // n_time_step * mini_batch_size, created in memory
    int n_time_step;
    int i_time_step; // time in rnn.
};

void math21_ml_function_dropout_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      mlfunction_node *finput, m21list *options);

void math21_ml_function_dropout_log(const mlfunction_dropout *f, const char *varName);

mlfunction_dropout *
math21_ml_function_dropout_create(mlfunction_node *fnode, mlfunction_node *finput, float rate, int n_time_step, const char *name);

void math21_ml_function_dropout_forward(mlfunction_dropout *f, mlfunction_node *finput,
                                        int is_train);

void math21_ml_function_dropout_backward(mlfunction_dropout *f, mlfunction_node *finput);

void math21_ml_function_dropout_saveState(const mlfunction_dropout *f, FILE *file);

void math21_ml_function_dropout_increase_by_time(mlfunction_dropout *f, int time_steps);

void math21_ml_function_dropout_reset(mlfunction_dropout *f);

#ifdef __cplusplus
}
#endif
