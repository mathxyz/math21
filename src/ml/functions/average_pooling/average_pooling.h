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

typedef struct mlfunction_average_pooling mlfunction_average_pooling;
struct mlfunction_average_pooling {
    const char* name;
    int batch; // mini_batch_size
    int h;
    int w;
    int c;
    int out_h;
    int out_w;
    int out_c;
    int outputs; // l.out_h*l.out_w*l.out_c;
    int inputs; // l.out_h*l.out_w*l.out_c;
    PointerFloatWrapper output; // Y
    PointerFloatWrapper delta; // dL/dY
};

void math21_ml_function_average_pooling_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                              const mlfunction_node *finput, m21list *options);

void math21_ml_function_average_pooling_log(const mlfunction_average_pooling *f, const char *varName);

mlfunction_average_pooling *
math21_ml_function_average_pooling_create(mlfunction_node *fnode, int batch, int c, int h, int w);

void math21_ml_function_average_pooling_resize(mlfunction_node *fnode, mlfunction_average_pooling *l, int w, int h);

void math21_ml_function_average_pooling_forward(mlfunction_average_pooling *f, const mlfunction_net *net, const mlfunction_node *finput);

void math21_ml_function_average_pooling_backward(mlfunction_average_pooling *f, mlfunction_node *finput);

void math21_ml_function_average_pooling_saveState(const mlfunction_average_pooling *f, FILE *file);

#ifdef __cplusplus
}
#endif
