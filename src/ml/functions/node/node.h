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

typedef enum {
    mlfnode_type_none,
    mlfnode_type_fully_connected,
    mlfnode_type_locally_connected,
    mlfnode_type_conv,
    mlfnode_type_deconv,
    mlfnode_type_batchnorm,
    mlfnode_type_average_pooling,
    mlfnode_type_max_pooling,
    mlfnode_type_cluster_pooling,
    mlfnode_type_softmax,
    mlfnode_type_sample,
    mlfnode_type_res,
    mlfnode_type_route,
    mlfnode_type_yolo,
    mlfnode_type_cost,
    mlfnode_type_rnn,
    mlfnode_type_gru,
    mlfnode_type_lstm,
    mlfnode_type_net,
    mlfnode_type_dropout,
//    mlfnode_type_time_distributed,
} mlfnode_type;

typedef struct mlfunction_node mlfunction_node;

struct mlfunction_net;
typedef struct mlfunction_net mlfunction_net;

// y = f(x), often wrapper function
struct mlfunction_node {
    int id; // id in net, or the number set to debug.
    mlfnode_type type; // id in net, or the number set to debug.
    int mini_batch_size; // It will contain time if there is time. It is total_mbs if total_mbs exists.
    int x_dim[3]; // can leave out, optional
    int y_dim[3]; // must set.
    int x_size; // x_size, no batch, may be not equal to x_dim[0] * x_dim[1] * x_dim[2];
    int y_size; // y_size, no batch
    PointerFloatWrapper dy; // dL/dY
    PointerFloatWrapper y; // Y
    void *function;

    void (*restoreState)(mlfunction_node *fnode, FILE *file);

    void (*saveState)(const mlfunction_node *fnode, FILE *file);

    size_t (*getGlobalSpaceSize)(mlfunction_node *fnode);

    // cost with mbs
    float (*getCost)(mlfunction_node *fnode);

    // todo: may deprecate this. checked once.
    // If deprecated, can it use `mlfunction_node *finput` as mbs info?
    // If mbs = 1, then time in rnn will have to be 1.
    void (*set_mbs)(mlfunction_node *fnode, int mini_batch_size);

    // non-constant pointer
    void (*forward)(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput);

    void (*backward)(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput);

    void (*update)(mlfunction_node *fnode, OptUpdate *optUpdate);

    void (*log)(const mlfunction_node *fnode, const char *varName);

    // deprecate
    const void *(*getDataToCpu)(mlfunction_node *fnode, const char *varName);

    m21rawtensor (*getRawTensorToCpu)(mlfunction_node *fnode, const char *varName);

    const char *(*getName)(const mlfunction_node *fnode);

    int stopbackward;
    int dontsave;
    int dontload;
};

mlfunction_node *math21_ml_function_node_create();

// user should set node to 0 after the call.
void math21_ml_function_node_destroy(mlfunction_node *node);

void math21_ml_function_node_log(const mlfunction_node *node, const char *name);

mlfnode_type math21_ml_function_node_type_string_to_type(const char *type);

#ifdef __cplusplus
}
#endif
