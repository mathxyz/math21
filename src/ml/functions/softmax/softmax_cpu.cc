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
#include "softmax_cpu.h"

#ifdef MATH21_FLAG_USE_CPU
void math21_ml_function_softmax_tree_cpu(float *input, int in_class_size, int mini_batch_size, int stride, float temp, float *output, m21tree hier)
{
    math21_tool_assert(0);
}

void softmax(float *input, int n, float temp, int stride, float *output) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i * stride] > largest) largest = input[i * stride];
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i * stride] /= sum;
    }
}


void math21_ml_function_softmax_cpu(float *input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride,
                 float temp, float *output) {
    int g, b;
    for (b = 0; b < mini_batch_size; ++b) {
        for (g = 0; g < groups; ++g) {
            softmax(input + b * batch_offset + g * group_offset, n, temp, stride,
                    output + b * batch_offset + g * group_offset);
        }
    }
}

void math21_ml_function_softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t - p;
    }
}
#endif