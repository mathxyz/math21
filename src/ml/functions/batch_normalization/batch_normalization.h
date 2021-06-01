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

typedef struct mlfunction_batchnorm mlfunction_batchnorm;

// use _cpu only when not cpu.
struct mlfunction_batchnorm {
    const char* name;
    int is_this_type;
    int is_train; // is train
    int outputs; // y_size, no batch
    int mini_batch_size; // mini_batch_size
    PointerFloatWrapper output; // Y
    PointerFloatWrapper x; // ?
//    int features_size;
    int out_c; // features_size
    int in_class_size;
    PointerFloatWrapper mean;
    PointerFloatWrapper variance;
    PointerFloatWrapper mean_delta;
    PointerFloatWrapper variance_delta;
    PointerFloatWrapper rolling_mean;
    float *rolling_mean_cpu;
    PointerFloatWrapper rolling_variance;
    float *rolling_variance_cpu;
    PointerFloatWrapper x_norm;
    PointerFloatWrapper biases; // b
    float *biases_cpu; // when not cpu
    PointerFloatWrapper bias_updates; // dL/db
    PointerFloatWrapper scales;
    float *scales_cpu;
    PointerFloatWrapper scale_updates;

    PointerFloatWrapper delta; // dL/dY. If not owned, it will not be reset when forward in train.
    int out_h, out_w; // nr_Y, nc_Y, nch_Y
    int h, w, c; // nr_X, nc_X, nch_X
    int inputs; // x_size, no batch

    // adam
    PointerFloatWrapper bias_m;
    PointerFloatWrapper bias_v;
    PointerFloatWrapper scale_m;
    PointerFloatWrapper scale_v;

    float learning_rate_scale;

    int total_mbs; // n_time_step * mini_batch_size, created in memory
    int n_time_step;
    int i_time_step; // time in rnn
};

void math21_ml_function_batchnorm_parse(mlfunction_node *fnode, const mlfunction_net *net,
                                        const mlfunction_node *finput, m21list *options);

void math21_ml_function_batchnorm_save_theta(mlfunction_batchnorm *l, FILE *fp, int isPull);

void math21_ml_function_batchnorm_load_theta(mlfunction_batchnorm *l, FILE *fp, int isPush);

void math21_ml_function_batchnorm_log(const mlfunction_batchnorm *f, const char *varName);

mlfunction_batchnorm *math21_ml_function_batchnorm_create(
        mlfunction_node *fnode, int is_this_type, mlfunction_node *finput,
        int mini_batch_size, int nc_Y, int nr_Y, int nch_Y, int adam);

void math21_ml_function_batchnorm_resize(mlfunction_batchnorm *l, mlfunction_node *fnode, int nc_Y, int nr_Y);

void math21_ml_function_batchnorm_forward(mlfunction_batchnorm *l, mlfunction_node *fnode);

void math21_ml_function_batchnorm_backward(mlfunction_batchnorm *l, mlfunction_node *fnode);

void math21_ml_function_batchnorm_update(mlfunction_batchnorm *l, OptUpdate *optUpdate);

void math21_ml_batch_normalization_scale_backward(PointerFloatInputWrapper X, PointerFloatInputWrapper dY, int mini_batch_size, int features_size,
                                                  int in_class_size, PointerFloatWrapper dk);

void math21_ml_batchnormalization_backward_mu_wrapper(PointerFloatInputWrapper dX_hat, PointerFloatInputWrapper variance, int mini_batch_size,
                                                      int features_size, int in_class_size, PointerFloatWrapper dmu);

void math21_ml_batchnormalization_backward_sigma_square_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper dX_hat, PointerFloatInputWrapper mu,
                                                                PointerFloatInputWrapper variance, int mini_batch_size,
                                                                int features_size, int in_class_size, PointerFloatWrapper dvariance);

void math21_ml_batchnormalization_backward_input_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper mu, PointerFloatInputWrapper variance,
                                                         PointerFloatInputWrapper dmu, PointerFloatInputWrapper dvariance, int mini_batch_size,
                                                         int features_size, int in_class_size, PointerFloatWrapper dX_hat);

void math21_ml_function_batchnorm_saveState(const mlfunction_batchnorm *f, FILE *file);

void math21_ml_function_batchnorm_merge_to(mlfunction_batchnorm *f, mlfunction_batchnorm *fb);

void math21_ml_function_batchnorm_scale(mlfunction_batchnorm *f, float s);

void math21_ml_function_batchnorm_pull_wrapper(mlfunction_batchnorm *l, NumB useRolling);

void math21_ml_function_batchnorm_push_wrapper(mlfunction_batchnorm *l, NumB useRolling);

void math21_ml_function_batchnorm_push_by_wrapper(mlfunction_batchnorm *f, mlfunction_batchnorm *fb, NumB useRolling);

void math21_ml_function_batchnorm_increase_by_time(mlfunction_batchnorm *f, int time_steps);

void math21_ml_function_batchnorm_reset(mlfunction_batchnorm *f);

#ifdef __cplusplus
}
#endif
