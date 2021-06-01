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

typedef struct mlfunction_sample mlfunction_sample;

// y = f(x)
struct mlfunction_sample {
    const char* name;
    int batch; // mini_batch_size
    int stride; // stride
    int h, w, c; // nr_X, nc_X, nch_X
    int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y
    int reverse; // 0: upsample, 1: sumdownsample
    float scale;
};

void math21_ml_function_sample_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                     const mlfunction_node *finput, m21list *options);

void math21_ml_function_sample_log(const mlfunction_sample *f, const char *varName);

mlfunction_sample *math21_ml_function_sample_create(mlfunction_node *fnode, int mini_batch_size, int nc, int nr, int nch, int stride);

void math21_ml_function_sample_resize(mlfunction_node *fnode, mlfunction_sample *l, int nc, int nr);

void math21_ml_function_sample_forward(mlfunction_sample *l, mlfunction_node *finput, NumB is_train);

void math21_ml_function_sample_backward(mlfunction_sample *l, mlfunction_node *finput);

void math21_ml_function_sample_saveState(const mlfunction_sample *f, FILE *file);


#ifdef __cplusplus
}
#endif
