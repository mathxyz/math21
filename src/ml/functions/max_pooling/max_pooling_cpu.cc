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
#include "max_pooling_cpu.h"

#ifdef MATH21_FLAG_USE_CPU
void math21_ml_function_max_pooling_forward_cpu(mlfunction_max_pooling *f, const mlfunction_node *finput) {
    int b, i, j, k, m, n;
    int w_offset = -f->padding / 2;
    int h_offset = -f->padding / 2;

    int h = f->out_h;
    int w = f->out_w;
    int c = f->c;

    for (b = 0; b < f->batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < f->size; ++n) {
                        for (m = 0; m < f->size; ++m) {
                            int cur_h = h_offset + i * f->stride + n;
                            int cur_w = w_offset + j * f->stride + m;
                            int index = cur_w + f->w * (cur_h + f->h * (k + b * f->c));
                            int valid = (cur_h >= 0 && cur_h < f->h &&
                                         cur_w >= 0 && cur_w < f->w);
                            float val = (valid != 0) ? finput->y[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    f->output[out_index] = max;
                    f->indexes[out_index] = max_i;
                }
            }
        }
    }
}

void math21_ml_function_max_pooling_backward_cpu(mlfunction_max_pooling *f, mlfunction_node *finput) {
    int i;
    int h = f->out_h;
    int w = f->out_w;
    int c = f->c;
    for (i = 0; i < h * w * c * f->batch; ++i) {
        int index = f->indexes[i];
        finput->dy[index] += f->delta[i];
    }
}
#endif