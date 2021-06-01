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
#include "../batch_normalization/files_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlfunction_conv mlfunction_conv;

// y = f(x)
struct mlfunction_conv {
    const char* name;
    int h, w, c; // nr_X, nc_X, nch_X
    int out_h, out_w, out_c; // nr_Y, nc_Y, nch_Y
    int groups; // num_group. Different groups share same W.
    int n; // nch_Y, num_box
    int binary;
    int xnor;
    int batch; // mini_batch_size
    int stride; // stride
    int size; // k_size
    int pad; // padding
    int inputs; // x_size, no batch
    int outputs; // y_size, no batch
    int nweights; // n_W
    PointerFloatWrapper weights; // W
    float *weights_cpu; // W
    PointerFloatWrapper weight_updates; // dL/dW
    PointerFloatWrapper biases; // b
    float *biases_cpu; // b
    PointerFloatWrapper bias_updates; // dL/db
    PointerFloatWrapper delta; // dL/dY
    PointerFloatWrapper output; // Y
    size_t workspace_size; // X_prime_size in convolution. shape: (k1_size*k2_size*nch_X)*(nr_Y*nc_Y)
    MATH21_FUNCTION_ACTIVATION_TYPE activation;
    mlfunction_batchnorm *bn;

    // adam
    PointerFloatWrapper m;
    PointerFloatWrapper v;
    PointerFloatWrapper bias_m;
    PointerFloatWrapper bias_v;

    // binary
    PointerFloatWrapper binary_weights;
    PointerFloatWrapper binary_input;

    float clip; // constant k, so -k<= x <= k, used only clip != 0. default k = 0
    float learning_rate_scale;

    float smooth;
    int flipped; // transpose, used in load weight.
    void *detail;
    mlfunction_node *fnode; // for debug only
};

void math21_ml_function_conv_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options);

void math21_ml_function_conv_save_theta(mlfunction_conv *l, FILE *fp);

void math21_ml_function_conv_load_theta(mlfunction_conv *l, FILE *fp);

void math21_ml_function_conv_log(const mlfunction_conv *f, const char *varName);

const void *math21_ml_function_conv_getDataToCpu(mlfunction_conv *f, const char *varName);

m21rawtensor math21_ml_function_conv_getRawTensorToCpu(mlfunction_conv *f, const char *varName);

mlfunction_conv *
math21_ml_function_conv_create(mlfunction_node *fnode, int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
                               int num_group,
                               int k_size, int stride, int padding, MATH21_FUNCTION_ACTIVATION_TYPE activation,
                               int is_batch_normalize,
                               int binary, int xnor, int adam);

void math21_ml_function_conv_resize(mlfunction_node *fnode, mlfunction_conv *l, int nc_X, int nr_X);

int math21_ml_function_conv_cal_nr_or_nc_Y(int nr_X, int pad, int size, int stride);

void
math21_ml_function_conv_bias_backward(PointerFloatWrapper db, PointerFloatInputWrapper dY, int mini_batch_size, int features_size,
                                      int in_class_size);

void math21_ml_function_conv_update(mlfunction_conv *l, OptUpdate *optUpdate);

void math21_ml_function_conv_swap_binary_weights(mlfunction_conv *l);

void math21_ml_function_conv_forward(mlfunction_conv *l, const mlfunction_node *finput,
                                     int is_train, PointerFloatWrapper workspace);

void math21_ml_function_conv_backward(mlfunction_conv *l, mlfunction_node *finput,
                                      int is_train, PointerFloatWrapper workspace);

void math21_ml_function_conv_saveState(const mlfunction_conv *f, FILE *file);

void math21_ml_function_conv_merge_to(mlfunction_conv *f, mlfunction_conv *fb);

void math21_ml_function_conv_scale(mlfunction_conv *f, float s);

void math21_ml_function_conv_pull_wrapper(mlfunction_conv *l, NumB useRolling);

void math21_ml_function_conv_push_wrapper(mlfunction_conv *l, NumB useRolling);

void math21_ml_function_conv_push_by_wrapper(mlfunction_conv *f, mlfunction_conv *fb, NumB useRolling);

void math21_ml_function_conv_set_mbs(mlfunction_conv *f, int mini_batch_size);

#ifdef __cplusplus
}
#endif
