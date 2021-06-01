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

typedef struct mlfunction_softmax mlfunction_softmax;
struct mlfunction_softmax {
    const char* name;
    int batch; // mini_batch_size
    int groups; // num_group. Different groups share same W.
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    PointerFloatWrapper loss;
    float *loss_cpu;
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y
    float *cost;

    m21tree *softmax_tree;
    float temperature;
    int spatial;
    int noloss;
    int h, w, c; // nr_X, nc_X, nch_X
};

void math21_ml_function_softmax_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      const mlfunction_node *finput, m21list *options);

void math21_ml_function_softmax_log(const mlfunction_softmax *f, const char *varName);

mlfunction_softmax *math21_ml_function_softmax_create(mlfunction_node *fnode, const mlfunction_node *finput, int groups);

void math21_ml_function_softmax_forward(mlfunction_softmax *f, mlfunction_net *net, mlfunction_node* finput);

void math21_ml_function_softmax_backward(mlfunction_softmax *f, mlfunction_net *net, mlfunction_node* finput);

void math21_ml_function_softmax_saveState(const mlfunction_softmax *f, FILE *file);

void math21_ml_function_softmax_net_set_temperature(mlfunction_net *fnet, float t);

#ifdef __cplusplus
}
#endif
