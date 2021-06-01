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

#include "conv_cpu.h"
#ifdef MATH21_FLAG_USE_CPU
float math21_ml_function_conv_X_to_X_prime_get_value(const float *X, int nr_X, int nc_X, int nch_X,
                                                     int ir_X, int ic_X, int ich_X, int pad) {
    ir_X -= pad;
    ic_X -= pad;

    if (ir_X < 0 || ic_X < 0 ||
        ir_X >= nr_X || ic_X >= nc_X)
        return 0;
    int index = (ich_X * nr_X + ir_X) * nc_X + ic_X;
    return X[index];
}

// X -> X_prime
// n_common = l.size*l.size*l.c/l.groups
// X_prime shape: (nch_X * nr_K * nc_K ) * nc_Y_m
// X_prime shape: (nch_K * nr_K * nc_K ) * nc_Y_m
// X_prime shape: n_common * nc_Y_m
// X_prime shape: n_common * (nc_X_prime_1 * nc_X_prime_2)
// X_prime shape: nr_X_prime * (nc_X_prime_1 * nc_X_prime_2)
// X_prime size: (nch_X * nr_K * nc_K ) * (nc_X_prime_1 * nc_X_prime_2)
void math21_ml_function_conv_X_to_X_prime_cpu(const float *X,
                                              int nch_X, int nr_X, int nc_X,
                                              int ksize, int stride, int pad, float *X_prime) {
    int ir, ic1, ic2;
    int nc_X_prime_1 = (nr_X + 2 * pad - ksize) / stride + 1;
    int nc_X_prime_2 = (nc_X + 2 * pad - ksize) / stride + 1;

    int nr_X_prime = nch_X * ksize * ksize;
    for (ir = 0; ir < nr_X_prime; ++ir) {
        int ic_K = ir % ksize;
        int ir_K = (ir / ksize) % ksize;
        int ich_X = ir / ksize / ksize; // ich_X = ich_K
        for (ic1 = 0; ic1 < nc_X_prime_1; ++ic1) {
            for (ic2 = 0; ic2 < nc_X_prime_2; ++ic2) {
                int index_X_prime = (ir * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2;
                int ir_X = ir_K + ic1 * stride;
                int ic_X = ic_K + ic2 * stride;
                X_prime[index_X_prime] = math21_ml_function_conv_X_to_X_prime_get_value(X, nr_X, nc_X, nch_X,
                                                                                        ir_X, ic_X, ich_X, pad);
            }
        }
    }
}

void math21_ml_function_conv_binarize_weights_cpu(const float *weights, int features_size, int size, float *binary) {
    int i, ifeature;
    for (ifeature = 0; ifeature < features_size; ++ifeature) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[ifeature * size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[ifeature * size + i] = (weights[ifeature * size + i] > 0) ? mean : -mean;
        }
    }
}

void math21_ml_function_conv_binarize_input_cpu(const float *input, int n, float *binary) {
    int i;
    for (i = 0; i < n; ++i) {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void math21_ml_function_conv_dX_prime_to_dX_add_value_cpu(float *X, int nr, int nc, int nch,
                                                          int ir, int ic, int ich, int pad, float val) {
    ir -= pad;
    ic -= pad;

    if (ir < 0 || ic < 0 ||
        ir >= nr || ic >= nc)
        return;
    X[ic + nc * (ir + nr * ich)] += val;
}

// dX_prime -> dX, dX_prime: n_common*nc_Y_m
// n_common = l.size*l.size*l.c/l.groups
void math21_ml_function_conv_dX_prime_to_dX_cpu(const float *dX_prime,
                                                int nch_X, int nr_X, int nc_X,
                                                int ksize, int stride, int pad, float *dX) {
    int ir, ic1, ic2;
    int nc_X_prime_1 = (nr_X + 2 * pad - ksize) / stride + 1;
    int nc_X_prime_2 = (nc_X + 2 * pad - ksize) / stride + 1;

    int nr_X_prime = nch_X * ksize * ksize;
    for (ir = 0; ir < nr_X_prime; ++ir) {
        int ir_K = (ir / ksize) % ksize;
        int ic_K = ir % ksize;
        int ich_X = ir / ksize / ksize; // ich_X = ich_K
        for (ic1 = 0; ic1 < nc_X_prime_1; ++ic1) {
            for (ic2 = 0; ic2 < nc_X_prime_2; ++ic2) {
                int ir_X = ir_K + ic1 * stride;
                int ic_X = ic_K + ic2 * stride;
                int index_X_prime = (ir * nc_X_prime_1 + ic1) * nc_X_prime_2 + ic2;
                double val = dX_prime[index_X_prime];
                math21_ml_function_conv_dX_prime_to_dX_add_value_cpu(dX, nr_X, nc_X, nch_X,
                                                                     ir_X, ic_X, ich_X, pad, val);
            }
        }
    }
}
#endif