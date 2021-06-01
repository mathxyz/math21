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

#include "update_cpu.h"

// todo: optimize
// alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t),
// eps_hat, see tensorflow/python/training/adam.py
void math21_optimization_adam_update_part_2_cpu(int x_size, float *x, float *m, float *v, float beta1, float beta2,
                                                float alpha, float eps, int t) {
    int i;
    for (i = 0; i < x_size; ++i) {
        // compute bias-corrected first moment estimate
        float mhat = m[i] / (1.f - powf(beta1, t));
        // compute bias-corrected second raw moment estimate
        float vhat = v[i] / (1.f - powf(beta2, t));

        // update
        // x = x - alpha * m / (sqrt(v) + eps)
        x[i] = x[i] + alpha * mhat / (sqrtf(vhat) + eps);
    }
}
