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
#include "../net/files_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlfunction_route mlfunction_route;

struct mlfunction_route {
    const char* name;
    int mini_batch_size;
    int num_layer;
    int *input_layers;
    int *input_sizes;
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
    PointerFloatWrapper output; // Y, with batch
    PointerFloatWrapper delta; // dL/dY
};

void math21_ml_function_route_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                    const mlfunction_node *finput, m21list *options);

void math21_ml_function_route_log(const mlfunction_route *f, const char *varName);

mlfunction_route *math21_ml_function_route_create(mlfunction_node *fnode, const mlfunction_net *net, int mini_batch_size, int num_layer, int *input_layers);

void math21_ml_function_route_resize(mlfunction_node *fnode, mlfunction_route *l, const mlfunction_net *net);

void math21_ml_function_route_forward(mlfunction_route *l, mlfunction_net *net);

void math21_ml_function_route_backward(mlfunction_route *l, mlfunction_net *net);

void math21_ml_function_route_saveState(const mlfunction_route *f, FILE *file);

#ifdef __cplusplus
}
#endif
