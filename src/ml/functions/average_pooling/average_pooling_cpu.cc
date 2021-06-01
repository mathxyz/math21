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

#include <float.h>
#include "average_pooling_cpu.h"

#ifdef MATH21_FLAG_USE_CPU
void math21_ml_function_average_pooling_forward_cpu(mlfunction_average_pooling *f, const mlfunction_node *finput) {
    int b, i, k;

    for (b = 0; b < f->batch; ++b) {
        for (k = 0; k < f->c; ++k) {
            int out_index = k + b * f->c;
            f->output[out_index] = 0;
            for (i = 0; i < f->h * f->w; ++i) {
                int in_index = i + f->h * f->w * (k + b * f->c);
                f->output[out_index] += finput->y[in_index];
            }
            f->output[out_index] /= f->h * f->w;
        }
    }
}

void math21_ml_function_average_pooling_backward_cpu(mlfunction_average_pooling *f, mlfunction_node *finput) {
    int b, i, k;

    for (b = 0; b < f->batch; ++b) {
        for (k = 0; k < f->c; ++k) {
            int out_index = k + b * f->c;
            for (i = 0; i < f->h * f->w; ++i) {
                int in_index = i + f->h * f->w * (k + b * f->c);
                finput->dy[in_index] += f->delta[out_index] / (f->h * f->w);
            }
        }
    }
}
#endif