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

typedef struct mlfunction_res mlfunction_res;

// y = f(x)
struct mlfunction_res {
    const char* name;
    int batch; // mini_batch_size
    int h, w, c; // nr_X, nc_X, nch_X
    int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int index;
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y

    float k1, k2;
    MATH21_FUNCTION_ACTIVATION_TYPE activation;
};

void math21_ml_function_res_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options);

void math21_ml_function_res_log(const mlfunction_res *f, const char *varName);

// shortcut
mlfunction_res *math21_ml_function_res_create(mlfunction_node *fnode, int batch, int index, int w, int h, int c, int w2, int h2, int c2);

void math21_ml_function_res_resize(mlfunction_node *fnode, mlfunction_res *l, int w, int h);

void math21_ml_function_res_forward(mlfunction_res *l, mlfunction_net *net, mlfunction_node *finput);

void math21_ml_function_res_backward(mlfunction_res *l, mlfunction_net *net, mlfunction_node *finput);

void math21_ml_function_res_saveState(const mlfunction_res *f, FILE *file);

#ifdef __cplusplus
}
#endif
