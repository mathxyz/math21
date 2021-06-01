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
    mlfnode_cost_type_L2,
    mlfnode_cost_type_masked,
    mlfnode_cost_type_L1,
    mlfnode_cost_type_seg,
    mlfnode_cost_type_smooth,
    mlfnode_cost_type_wgan,
} mlfnode_cost_type;

typedef struct mlfunction_cost mlfunction_cost;
struct mlfunction_cost {
    const char* name;
    int batch; // mini_batch_size
    int groups; // num_group. Different groups share same W.
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int y_dim[MATH21_DIMS_RAW_TENSOR];
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y
    float * tmp_cpu; // Y
    float *cost;
    mlfnode_cost_type cost_type;
    float scale;
    float ratio;
    float noobject_scale;
    float thresh;
    float smooth;
};

mlfnode_cost_type math21_ml_function_cost_type_get_type(const char *s);

const char *math21_ml_function_cost_type_get_name(mlfnode_cost_type a);

void math21_ml_function_cost_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options);

void math21_ml_function_cost_log(const mlfunction_cost *f, const char *varName);

mlfunction_cost *math21_ml_function_cost_create(
        mlfunction_node *fnode, const mlfunction_node *finput,
        mlfnode_cost_type cost_type, float scale);

void math21_ml_function_cost_resize(mlfunction_node *fnode, mlfunction_cost *l, int w, int h);

void math21_ml_function_cost_forward(mlfunction_cost *f, mlfunction_net *net, mlfunction_node *finput);

void math21_ml_function_cost_backward(mlfunction_cost *f, mlfunction_net *net, mlfunction_node *finput);

void math21_ml_function_cost_saveState(const mlfunction_cost *f, FILE *file);

#ifdef __cplusplus
}
#endif
