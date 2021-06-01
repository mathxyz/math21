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

#include "dropout_cpu.h"

#ifdef MATH21_FLAG_USE_CPU
void math21_ml_function_dropout_forward_cpu(mlfunction_dropout *f, mlfunction_node *finput, int is_train) {
    int i;
    if (!is_train) return;
    int size = f->inputs * f->batch;
    if (f->i_time_step==0) {
        math21_vector_pr_rand_uniform_01_wrapper(f->rand, size);
    }

    for (i = 0; i < size; ++i) {
        float r = f->rand[i];
        if (r < f->rate) f->y[i] = 0;
        else f->y[i] = f->scale * finput->y[i]; // see nn_ops.dropout_v2 in tensorflow
    }
}

void math21_ml_function_dropout_backward_cpu(mlfunction_dropout *f, mlfunction_node *finput) {
    int i;
    if (!finput->dy) return;
    int size = f->inputs * f->batch;
    for (i = 0; i < size; ++i) {
        float r = f->rand[i];
        if (r < f->rate) {}
        else finput->dy[i] += f->scale * f->dy[i];
    }
}

#endif