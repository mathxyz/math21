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

#include <stdio.h>
#include <assert.h>
#include "batch_normalization_cpu.h"
#include "batch_normalization_cuda.h"
#include "batch_normalization_opencl.h"
#include "batch_normalization.h"
#include "../conv/files_c.h"
#include "inner_cc.h"

void math21_ml_function_batchnorm_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                        const mlfunction_node *finput, m21list *options) {
    math21_ml_function_batchnorm_create(fnode, 1, 0, finput->mini_batch_size, finput->y_dim[2],
                                        finput->y_dim[1], finput->y_dim[0], 0);
}

void math21_ml_function_batchnorm_save_theta(mlfunction_batchnorm *f, FILE *fp, int isPull) {
#ifndef MATH21_FLAG_USE_CPU
    if (isPull) {
        math21_ml_function_batchnorm_pull_wrapper(f, 1);
    }
#endif

#if defined(MATH21_FLAG_USE_CPU)
    float * biases = f->biases;
    float * scales = f->scales;
    float * rolling_mean = f->rolling_mean;
    float * rolling_variance = f->rolling_variance;
#else
    float *biases = f->biases_cpu;
    float *scales = f->scales_cpu;
    float *rolling_mean = f->rolling_mean_cpu;
    float *rolling_variance = f->rolling_variance_cpu;
#endif
    fwrite(biases, sizeof(float), f->out_c, fp);
    fwrite(scales, sizeof(float), f->out_c, fp);
    fwrite(rolling_mean, sizeof(float), f->out_c, fp);
    fwrite(rolling_variance, sizeof(float), f->out_c, fp);
}

void math21_ml_function_batchnorm_load_theta(mlfunction_batchnorm *f, FILE *fp, int isPush) {
#if defined(MATH21_FLAG_USE_CPU)
    float * biases = f->biases;
    float * scales = f->scales;
    float * rolling_mean = f->rolling_mean;
    float * rolling_variance = f->rolling_variance;
#else
    float *biases = f->biases_cpu;
    float *scales = f->scales_cpu;
    float *rolling_mean = f->rolling_mean_cpu;
    float *rolling_variance = f->rolling_variance_cpu;
#endif

    fread(biases, sizeof(float), f->out_c, fp);
    fread(scales, sizeof(float), f->out_c, fp);
    fread(rolling_mean, sizeof(float), f->out_c, fp);
    fread(rolling_variance, sizeof(float), f->out_c, fp);
#ifndef MATH21_FLAG_USE_CPU
    if (isPush) {
        math21_ml_function_batchnorm_push_wrapper(f, 1);
    }
#endif
}

void math21_ml_function_batchnorm_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_batchnorm *f = (mlfunction_batchnorm *) fnode->function;
    math21_ml_function_batchnorm_saveState(f, file);
}

void math21_ml_function_batchnorm_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_batchnorm *f = (mlfunction_batchnorm *) fnode->function;
    f->mini_batch_size = mini_batch_size;
}

void
math21_ml_function_batchnorm_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_batchnorm *) fnode->function;
    f->is_train = net->is_train;
    math21_ml_function_batchnorm_forward(f, finput);
}

void
math21_ml_function_batchnorm_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_batchnorm *f = (mlfunction_batchnorm *) fnode->function;
    f->is_train = net->is_train;
    math21_ml_function_batchnorm_backward(f, finput);
}

void math21_ml_function_batchnorm_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    mlfunction_batchnorm *f = (mlfunction_batchnorm *) fnode->function;
    math21_ml_function_batchnorm_update(f, optUpdate);
}

void math21_ml_function_batchnorm_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_batchnorm *) fnode->function;
    math21_ml_function_batchnorm_log(f, varName);
}

const char *math21_ml_function_batchnorm_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_batchnorm *) fnode->function;
    return f->name;
}


void math21_ml_function_batchnorm_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_batchnorm *) fnode->function;
    fnode->mini_batch_size = f->total_mbs;
    fnode->x_dim[0] = f->c;
    fnode->x_dim[1] = f->h;
    fnode->x_dim[2] = f->w;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->x_size = fnode->x_dim[0] * fnode->x_dim[1] * fnode->x_dim[2];
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

void math21_ml_function_batchnorm_log(const mlfunction_batchnorm *f, const char *varName) {
    if (f->is_this_type) {
        fprintf(stdout, "%s: %d x %d x %d\n", f->name, f->h, f->w, f->c);
    } else {
        fprintf(stdout, "not type %s\n", f->name);
    }
}

mlfunction_batchnorm *math21_ml_function_batchnorm_create(
        mlfunction_node *fnode, int is_this_type, mlfunction_node *finput,
        int mini_batch_size, int nc_Y, int nr_Y, int nch_Y, int adam) {
    auto *f = (mlfunction_batchnorm *) math21_vector_calloc_cpu(1, sizeof(mlfunction_batchnorm));
    f->is_this_type = is_this_type;
    f->mini_batch_size = mini_batch_size;
    f->h = f->out_h = nr_Y;
    f->w = f->out_w = nc_Y;
    f->c = f->out_c = nch_Y;
    f->inputs = nc_Y * nr_Y * nch_Y;
    f->outputs = f->inputs;
    f->in_class_size = f->out_h * f->out_w;

    f->biases = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
    f->bias_updates = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

    f->scales = math21_vector_create_with_default_value_wrapper(nch_Y, 1);
    f->scale_updates = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

    f->mean = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
    f->variance = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

    f->mean_delta = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
    f->variance_delta = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

    f->rolling_mean = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
    f->rolling_variance = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

    f->x = math21_vector_create_with_default_value_wrapper(f->mini_batch_size * f->outputs, 0);
    f->x_norm = math21_vector_create_with_default_value_wrapper(f->mini_batch_size * f->outputs, 0);

#ifndef MATH21_FLAG_USE_CPU
    f->biases_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 0);
    f->scales_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 1);
    f->rolling_mean_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 0);
    f->rolling_variance_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 0);
#endif

    if (adam) {
        f->bias_m = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
        f->bias_v = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
        f->scale_m = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
        f->scale_v = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
    }

    if (is_this_type) {
        f->output = math21_vector_create_with_default_value_wrapper(nr_Y * nc_Y * nch_Y * f->mini_batch_size, 0);
        f->delta = math21_vector_create_with_default_value_wrapper(nr_Y * nc_Y * nch_Y * f->mini_batch_size, 0);
    } else {
        math21_tool_assert(finput);
        f->output = finput->y;
        f->delta = finput->dy;
    }
    f->total_mbs = f->mini_batch_size;
    f->n_time_step = 1;
    f->i_time_step = 0;

    f->name = math21_string_create_from_string("batch normalization");
    if (fnode) {
        fnode->type = mlfnode_type_batchnorm;
        fnode->function = f;
        fnode->saveState = math21_ml_function_batchnorm_node_saveState;
        fnode->set_mbs = math21_ml_function_batchnorm_node_set_mbs;
        fnode->forward = math21_ml_function_batchnorm_node_forward;
        fnode->backward = math21_ml_function_batchnorm_node_backward;
        fnode->update = math21_ml_function_batchnorm_node_update;
        fnode->log = math21_ml_function_batchnorm_node_log;
        fnode->getName = math21_ml_function_batchnorm_node_getName;
        math21_ml_function_batchnorm_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_batchnorm_resize(mlfunction_batchnorm *f, mlfunction_node *fnode, int nc_Y, int nr_Y) {
    f->h = f->out_h = nr_Y;
    f->w = f->out_w = nc_Y;
    f->outputs = f->out_h * f->out_w * f->out_c;
    f->inputs = f->outputs;
    f->in_class_size = f->out_h * f->out_w;

    f->x = math21_vector_resize_with_default_value_wrapper(f->x, f->total_mbs * f->outputs, 0);
    f->x_norm = math21_vector_resize_with_default_value_wrapper(f->x_norm, f->total_mbs * f->outputs, 0);
    if (f->is_this_type) {
        f->delta = math21_vector_resize_with_default_value_wrapper(f->delta, f->total_mbs * f->outputs, 0);
        f->output = math21_vector_resize_with_default_value_wrapper(f->output, f->total_mbs * f->outputs, 0);
    } else {
        assert(fnode);
        f->output = fnode->y;
        f->delta = fnode->dy;
    }
    if (fnode) {
        if (!fnode->function) {
            fnode->function = f;
        }
        math21_ml_function_batchnorm_node_reset(fnode);
    }
}

// Y = BN(X)
void math21_ml_function_batchnorm_forward(mlfunction_batchnorm *f, mlfunction_node *finput) {
    if (f->is_train) {
        if (f->is_this_type) {
            if (f->i_time_step == 0) {
                math21_vector_set_wrapper(f->total_mbs * f->outputs, 0, f->delta, 1);
            }
        }
    }
    // not for fully connected layer
    if (f->is_this_type) {
        math21_vector_assign_from_vector_wrapper(f->outputs * f->mini_batch_size, finput->y, 1, f->output, 1);
    }
    // get X for what
    math21_vector_assign_from_vector_wrapper(f->outputs * f->mini_batch_size, f->output, 1, f->x, 1);
    if (f->is_train) {
        // mu = E(X) for rnn, cnn
        math21_vector_mean_wrapper(f->output, f->mini_batch_size, f->out_c, f->in_class_size, f->mean);
        // sigma_square = Var(X)
        math21_vector_variance_wrapper(f->output, f->mean, f->mini_batch_size, f->out_c, f->in_class_size, f->variance);

        // update rolling mean
        math21_vector_kx_wrapper(f->out_c, .99, f->rolling_mean, 1);
        math21_vector_kx_add_y_wrapper(f->out_c, .01, f->mean, 1, f->rolling_mean, 1);
        // update rolling variance
        math21_vector_kx_wrapper(f->out_c, .99, f->rolling_variance, 1);
        math21_vector_kx_add_y_wrapper(f->out_c, .01, f->variance, 1, f->rolling_variance, 1);


#ifndef MATH21_FLAG_USE_CPU
        // get X for what
        math21_vector_assign_from_vector_wrapper(f->outputs * f->mini_batch_size, f->output, 1, f->x, 1);
#endif

        // X_hat = (X-mu)/sigma
        math21_vector_normalize_wrapper(f->output, f->mean, f->variance, f->mini_batch_size, f->out_c,
                                        f->in_class_size);
        // get X_norm for what
        math21_vector_assign_from_vector_wrapper(f->outputs * f->mini_batch_size, f->output, 1, f->x_norm, 1);
    } else {
        math21_vector_normalize_wrapper(f->output, f->rolling_mean, f->rolling_variance, f->mini_batch_size, f->out_c,
                                        f->in_class_size);
    }

    // Y = scale * X_hat + bias
    math21_vector_kx_with_in_class_wrapper(f->output, f->scales, f->mini_batch_size, f->out_c, f->in_class_size);
    math21_vector_x_add_b_with_in_class_wrapper(f->output, f->biases, f->mini_batch_size, f->out_c, f->in_class_size);
}

void math21_ml_function_batchnorm_backward(mlfunction_batchnorm *f, mlfunction_node *finput) {
    if (!f->is_train) {
        math21_tool_assert(0 && "test!");
        f->mean = f->rolling_mean;
        f->variance = f->rolling_variance;
    }
    // dL/db += sum(dL/dY(i))
    math21_ml_function_conv_bias_backward(f->bias_updates, f->delta, f->mini_batch_size, f->out_c, f->out_w * f->out_h);
    // dL/dk += sum(dL/dY(i) *.ele X_hat(i))
    math21_ml_batch_normalization_scale_backward(f->x_norm, f->delta, f->mini_batch_size, f->out_c, f->out_w * f->out_h,
                                                 f->scale_updates);
    // dL/dX_hat = dL/dY * k
    math21_vector_kx_with_in_class_wrapper(f->delta, f->scales, f->mini_batch_size, f->out_c, f->out_h * f->out_w);
    // Todo: sum(dL/dX_hat(i) * dX_hat(i)/dmu) + dL/dsigma_square * dsigma_square/dmu
    math21_ml_batchnormalization_backward_mu_wrapper(f->delta, f->variance, f->mini_batch_size, f->out_c,
                                                     f->out_w * f->out_h,
                                                     f->mean_delta);
    // dL/dsigma_square = sum(dL/dX_hat(i) * dX_hat(i)/dsigma_square)
    math21_ml_batchnormalization_backward_sigma_square_wrapper(f->x, f->delta, f->mean, f->variance, f->mini_batch_size,
                                                               f->out_c, f->out_w * f->out_h, f->variance_delta);
    // dL/dX
    math21_ml_batchnormalization_backward_input_wrapper(f->x, f->mean, f->variance, f->mean_delta, f->variance_delta,
                                                        f->mini_batch_size, f->out_c, f->out_w * f->out_h, f->delta);
    if (f->is_this_type) {
        math21_vector_assign_from_vector_wrapper(f->outputs * f->mini_batch_size, f->delta, 1, finput->dy, 1);
    }
}

void math21_ml_function_batchnorm_update(mlfunction_batchnorm *f, OptUpdate *optUpdate) {
    OptUpdate_Adam *a = 0;
    if (optUpdate->type == OptUpdateType_Adam) {
        a = (OptUpdate_Adam *) optUpdate->detail;
    }
    float learning_rate = optUpdate->alpha * f->learning_rate_scale;
    float momentum = optUpdate->momentum;
    float decay = optUpdate->decay;
    int batch = optUpdate->mini_batch_size;
//    math21_tool_assert(f->total_mbs == batch && "check test");

    if (a) {
        math21_optimization_adam_update_wrapper(f->biases, f->bias_updates, f->bias_m, f->bias_v, a->beta1,
                                                a->beta2, a->eps, decay, learning_rate, f->out_c, batch, a->t);
        math21_optimization_adam_update_wrapper(f->scales, f->scale_updates, f->scale_m, f->scale_v,
                                                a->beta1, a->beta2, a->eps, decay, learning_rate, f->out_c, batch,
                                                a->t);
    } else {

        // b = b - alpha * dL/db
        // f->bias_updates = -dL/db because of loss function L is -L.
        float k = learning_rate / batch;
        math21_vector_kx_add_y_wrapper(f->out_c, k, f->bias_updates, 1, f->biases, 1);

        // dL/db = momentum * dL/db
        math21_vector_kx_wrapper(f->out_c, momentum, f->bias_updates, 1);

        // k = k - alpha * dL/dk
        math21_vector_kx_add_y_wrapper(f->out_c, learning_rate / batch, f->scale_updates, 1, f->scales, 1);
        // dL/dk = momentum * dL/dk
        math21_vector_kx_wrapper(f->out_c, momentum, f->scale_updates, 1);
    }
}


void math21_ml_batch_normalization_scale_backward(PointerFloatInputWrapper X, PointerFloatInputWrapper dY,
                                                  int mini_batch_size, int features_size,
                                                  int in_class_size, PointerFloatWrapper dk) {
    math21_vector_sum_SchurProduct_with_in_class_wrapper(X, dY, mini_batch_size, features_size, in_class_size, dk);
}

// error: dL/dmu = sum(dL/dX_hat(i) * dX_hat(i)/dmu)
// Todo: dL/dmu = sum(dL/dX_hat(i) * dX_hat(i)/dmu) + dL/dsigma_square * dsigma_square/dmu
void
math21_ml_batchnormalization_backward_mu_wrapper(PointerFloatInputWrapper dX_hat, PointerFloatInputWrapper variance,
                                                 int mini_batch_size,
                                                 int features_size, int in_class_size, PointerFloatWrapper dmu) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_batchnormalization_backward_mu_cpu(dX_hat, variance, mini_batch_size, features_size, in_class_size, dmu);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_batchnormalization_backward_mu_fast_cuda(dX_hat, variance, mini_batch_size, features_size, in_class_size,
                                                       dmu);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_batchnormalization_backward_mu_fast_opencl(dX_hat, variance, mini_batch_size, features_size,
                                                         in_class_size,
                                                         dmu);
#endif
}

void
math21_ml_batchnormalization_backward_sigma_square_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper dX_hat,
                                                           PointerFloatInputWrapper mu,
                                                           PointerFloatInputWrapper variance, int mini_batch_size,
                                                           int features_size, int in_class_size,
                                                           PointerFloatWrapper dvariance) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_batchnormalization_backward_sigma_square_cpu(X, dX_hat, mu, variance, mini_batch_size, features_size, in_class_size, dvariance);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_batchnormalization_backward_sigma_square_fast_cuda(X, dX_hat, mu, variance, mini_batch_size,
                                                                 features_size, in_class_size, dvariance);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_batchnormalization_backward_sigma_square_fast_opencl(X, dX_hat, mu, variance, mini_batch_size,
                                                                   features_size, in_class_size, dvariance);
#endif
}

void math21_ml_batchnormalization_backward_input_wrapper(PointerFloatInputWrapper X, PointerFloatInputWrapper mu,
                                                         PointerFloatInputWrapper variance,
                                                         PointerFloatInputWrapper dmu,
                                                         PointerFloatInputWrapper dvariance, int mini_batch_size,
                                                         int features_size, int in_class_size,
                                                         PointerFloatWrapper dX_hat) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_batchnormalization_backward_input_cpu(X, mu, variance, dmu, dvariance, mini_batch_size, features_size, in_class_size, dX_hat);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_batchnormalization_backward_input_cuda(X, mu, variance, dmu, dvariance, mini_batch_size, features_size,
                                                     in_class_size, dX_hat);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_batchnormalization_backward_input_opencl(X, mu, variance, dmu, dvariance, mini_batch_size, features_size,
                                                     in_class_size, dX_hat);
#endif
}

void math21_ml_function_batchnorm_saveState(const mlfunction_batchnorm *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->biases, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->scales, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->rolling_mean, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->rolling_variance, f->out_c);

    math21_vector_serialize_c_wrapper(file, f->bias_updates, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->scale_updates, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->mean_delta, f->out_c);
    math21_vector_serialize_c_wrapper(file, f->variance_delta, f->out_c);
}

void math21_ml_function_batchnorm_merge_to(mlfunction_batchnorm *f, mlfunction_batchnorm *fb) {
    math21_vector_kx_add_y_cpu(f->out_c, 1, f->biases_cpu, 1, fb->biases_cpu, 1);
    math21_vector_kx_add_y_cpu(f->out_c, 1, f->scales_cpu, 1, fb->scales_cpu, 1);
}

void math21_ml_function_batchnorm_scale(mlfunction_batchnorm *f, float s) {
    math21_vector_kx_cpu(f->out_c, s, f->biases_cpu, 1);
    math21_vector_kx_cpu(f->out_c, s, f->scales_cpu, 1);
}

void math21_ml_function_batchnorm_pull_wrapper(mlfunction_batchnorm *f, NumB useRolling) {
    math21_vector_pull_wrapper(f->biases, f->biases_cpu, f->out_c);
    math21_vector_pull_wrapper(f->scales, f->scales_cpu, f->out_c);
    if (useRolling) {// why?
        math21_vector_pull_wrapper(f->rolling_mean, f->rolling_mean_cpu, f->out_c);
        math21_vector_pull_wrapper(f->rolling_variance, f->rolling_variance_cpu, f->out_c);
    }
}

void math21_ml_function_batchnorm_push_wrapper(mlfunction_batchnorm *f, NumB useRolling) {
    math21_ml_function_batchnorm_push_by_wrapper(f, f, useRolling);
}

// f is pushed by fb
void math21_ml_function_batchnorm_push_by_wrapper(mlfunction_batchnorm *f, mlfunction_batchnorm *fb, NumB useRolling) {
    math21_vector_push_wrapper(f->biases, fb->biases_cpu, f->out_c);
    math21_vector_push_wrapper(f->scales, fb->scales_cpu, f->out_c);
    if (useRolling) {
        math21_vector_push_wrapper(f->rolling_mean, fb->rolling_mean_cpu, f->out_c);
        math21_vector_push_wrapper(f->rolling_variance, fb->rolling_variance_cpu, f->out_c);
    }
}

void math21_ml_function_batchnorm_increase_by_time(mlfunction_batchnorm *f, int time_steps) {
    f->i_time_step += time_steps;
    int num = f->outputs * f->mini_batch_size * time_steps;
    f->output += num;
    f->delta += num;
    f->x += num;
    f->x_norm += num;
}

void math21_ml_function_batchnorm_reset(mlfunction_batchnorm *f) {
    int num = f->outputs * f->mini_batch_size * f->i_time_step;
    f->output -= num;
    f->delta -= num;
    f->x -= num;
    f->x_norm -= num;
    f->i_time_step = 0;
}

