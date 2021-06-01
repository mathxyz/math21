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

#include "conv_wrapper.h"
#include "conv_cuda.h"
#include "detail.h"
#include "inner_cc.h"
#include "conv.h"

using namespace math21;

// todo: consider add channels_last and channels_first
// todo: rename to cross-correlated
void math21_ml_function_conv_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    int nch_out = math21_function_option_find_int(options, "filters", 1);
    int k_size = math21_function_option_find_int(options, "size", 1);
    int stride = math21_function_option_find_int(options, "stride", 1);
    // todo: padding type: valid or same
    int pad = math21_function_option_find_int_quiet(options, "pad", 0);
    int padding = math21_function_option_find_int_quiet(options, "padding", 0);
    int groups = math21_function_option_find_int_quiet(options, "groups", 1);
    if (pad) padding = k_size / 2;

    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);

    int mini_batch_size, nr, nc, nch;
    nch = finput->y_dim[0];
    nr = finput->y_dim[1];
    nc = finput->y_dim[2];
    mini_batch_size = finput->mini_batch_size;
    if (!(nr && nc && nch)) math21_error("Layer before convolutional layer must output image.");
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);
    int binary = math21_function_option_find_int_quiet(options, "binary", 0);
    int xnor = math21_function_option_find_int_quiet(options, "xnor", 0);

    mlfunction_conv *f = math21_ml_function_conv_create(fnode, mini_batch_size, nr, nc, nch, nch_out, groups,
                                                        k_size,
                                                        stride, padding, activation, is_batch_normalize, binary,
                                                        xnor, fnet->adam);

    f->smooth = math21_function_option_find_float_quiet(options, "smooth", 0);
    f->flipped = math21_function_option_find_int_quiet(options, "flipped", 0);
    f->learning_rate_scale = math21_function_option_find_float_quiet(options, "learning_rate", 1);
}

void math21_ml_function_conv_save_theta(mlfunction_conv *f, FILE *fp) {
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_conv_pull_wrapper(f, 1);
#endif

#if defined(MATH21_FLAG_USE_CPU)
    float * weights = f->weights;
    float * biases = f->biases;
#else
    float *weights = f->weights_cpu;
    float *biases = f->biases_cpu;
#endif

    int num = f->nweights;
    if (f->bn) {
        math21_ml_function_batchnorm_save_theta(f->bn, fp, 0);
    } else {
        fwrite(biases, sizeof(float), f->n, fp);
    }
    fwrite(weights, sizeof(float), num, fp);
}

void math21_ml_function_conv_load_theta(mlfunction_conv *f, FILE *fp) {
#if defined(MATH21_FLAG_USE_CPU)
    float * weights = f->weights;
    float * biases = f->biases;
#else
    float *weights = f->weights_cpu;
    float *biases = f->biases_cpu;
#endif

    int num = f->c / f->groups * f->n * f->size * f->size;
    if (f->bn) {
        math21_ml_function_batchnorm_load_theta(f->bn, fp, 0);
    } else {
        fread(biases, sizeof(float), f->n, fp);
    }
    fread(weights, sizeof(float), num, fp);
    if (f->flipped) {
        math21_matrix_transpose(weights, f->c * f->size * f->size, f->n);
    }
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_conv_push_wrapper(f, 1);
#endif
}


size_t _math21_ml_function_conv_get_X_prime_size(int nr_Y, int nc_Y,
                                                 int k_size, int nch_X, int num_group) {
    return (size_t) (nch_X / num_group * k_size * k_size) * (nr_Y * nc_Y) * sizeof(float);
}

void math21_ml_function_conv_node_create(mlfunction_node *fnode) {
}

void math21_ml_function_conv_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_conv *f = (mlfunction_conv *) fnode->function;
    math21_ml_function_conv_saveState(f, file);
}

size_t math21_ml_function_conv_node_getGlobalSpaceSize(mlfunction_node *fnode) {
    mlfunction_conv *f = (mlfunction_conv *) fnode->function;
    return f->workspace_size;
}

// todo: check relation with n_time_step;
void math21_ml_function_conv_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_conv *f = (mlfunction_conv *) fnode->function;
    math21_ml_function_conv_set_mbs(f, mini_batch_size);
}

void math21_ml_function_conv_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *fconv = (mlfunction_conv *) fnode->function;
    math21_ml_function_conv_forward(fconv, finput, net->is_train, net->workspace);
}

void math21_ml_function_conv_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_conv *fconv = (mlfunction_conv *) fnode->function;
    math21_ml_function_conv_backward(fconv, finput, net->is_train, net->workspace);
}

void math21_ml_function_conv_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    mlfunction_conv *f = (mlfunction_conv *) fnode->function;
    math21_ml_function_conv_update(f, optUpdate);
}

const void *math21_ml_function_conv_node_getDataToCpu(mlfunction_node *fnode, const char *varName) {
    auto *f = (mlfunction_conv *) fnode->function;
    return math21_ml_function_conv_getDataToCpu(f, varName);
}

m21rawtensor math21_ml_function_conv_node_getRawTensorToCpu(mlfunction_node *fnode, const char *varName) {
    auto *f = (mlfunction_conv *) fnode->function;
    return math21_ml_function_conv_getRawTensorToCpu(f, varName);
}

void math21_ml_function_conv_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_conv *) fnode->function;
    math21_ml_function_conv_log(f, varName);
}

const char *math21_ml_function_conv_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_conv *) fnode->function;
    return f->name;
}

void math21_ml_function_conv_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_conv *) fnode->function;
    fnode->mini_batch_size = f->batch;
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

void math21_ml_function_conv_destroy(mlfunction_conv *f) {

}

void math21_ml_function_conv_log(const mlfunction_conv *f, const char *varName) {
    auto *f_detail = (const mlfunction_conv_detail *) f->detail;
    std::string _varNameNew;
    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
        return;
    }
    varName = _varNameNew.c_str();

    if (math21_string_is_equal(varName, "summary")) {

//        fprintf(stdout, "conv  %5d x%2d x%2d x%4d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", f->out_c, f->size, f->size, f->c,
//                f->stride,
//                f->h, f->w, f->c, f->out_h, f->out_w, f->out_c,
//                (2.0 * f->n * f->size * f->size * f->c / f->groups * f->out_h * f->out_w) / 1000000000.);

        // K shape: nch_Y * group_size_X * nr_K * nc_K;
        // X shape: mbs * num_group * group_size_X * nr_X * nc_X
        if (f->groups == 1) {
            fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), "
                            "K shape = (%d, %d, %d, %d), s = %d, BFLOPs = %5.3f\n",
                    f->name,
                    f->h, f->w, f->c, f->batch,
                    f->out_h, f->out_w, f->out_c, f->batch,
                    f->out_c, f->size, f->size, f->c / f->groups,
                    f->stride,
                    (2.0 * f->n * f->size * f->size * f->c / f->groups * f->out_h * f->out_w) / 1000000000.);
        } else {
            fprintf(stdout, "%s with groups: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d), "
                            "K shape = (%d, %d, %d, %d), s = %d, BFLOPs = %5.3f\n",
                    f->name,
                    f->h, f->w, f->c / f->groups, f->groups, f->batch,
                    f->out_h, f->out_w, f->out_c / f->groups, f->groups, f->batch,
                    f->out_c, f->size, f->size, f->c / f->groups,
                    f->stride,
                    (2.0 * f->n * f->size * f->size * f->c / f->groups * f->out_h * f->out_w) / 1000000000.);
        }
        return;
    }
    fprintf(stdout, "%s:\n", f->name);
    std::string name = varName;
    m21variable *var;
    if (f_detail->vars.get(varName, var)) {
        var->log(varName);
    } else if (name == "b") {
        if (f->bn) {
            math21_ml_function_batchnorm_log(f->bn, varName);
        }
    } else if (name == "db") {
        if (f->bn) {
            math21_ml_function_batchnorm_log(f->bn, varName);
        }
    } else {
        m21log("no variable name ", varName);
    }
}

const void *math21_ml_function_conv_getDataToCpu(mlfunction_conv *f, const char *varName) {
    auto *f_detail = (mlfunction_conv_detail *) f->detail;
    std::string _varNameNew;
    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
        return 0;
    }
    varName = _varNameNew.c_str();
    fprintf(stdout, "%s:\n", f->name);
    m21variable *var;
    if (f_detail->vars.get(varName, var)) {
        const NumR32 *p = 0;
        var->getDataToCpu(p);
        m21log("p", (void *) p);
//        math21_tensor_4d_float_log_cpu(__FUNCTION__, p, 18, 256, 1, 1);
        return p;
    } else {
        m21log("no variable name ", varName);
        return 0;
    }
}

m21rawtensor math21_ml_function_conv_getRawTensorToCpu(mlfunction_conv *f, const char *varName) {
    auto *f_detail = (mlfunction_conv_detail *) f->detail;
    std::string _varNameNew;
    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
        m21rawtensor rawtensor = {0};
        return rawtensor;
    }
    varName = _varNameNew.c_str();
    fprintf(stdout, "%s:\n", f->name);
    m21variable *var;
    if (f_detail->vars.get(varName, var)) {
        m21rawtensor p = {0};
        var->getRawTensorToCpu(p);
//        math21_rawtensor_log_cpu(__FUNCTION__ , p);
        return p;
    } else {
        m21log("no variable name ", varName);
        m21rawtensor rawtensor = {0};
        return rawtensor;
    }
}

mlfunction_conv *
math21_ml_function_conv_create(mlfunction_node *fnode, int mini_batch_size, int nr_X, int nc_X, int nch_X, int nch_Y,
                               int num_group,
                               int k_size, int stride, int padding, MATH21_FUNCTION_ACTIVATION_TYPE activation,
                               int is_batch_normalize,
                               int binary, int xnor, int adam) {
    auto *f = (mlfunction_conv *) math21_vector_calloc_cpu(1, sizeof(mlfunction_conv));
    auto *f_detail = new mlfunction_conv_detail();
    f->detail = f_detail;

    f->c = nch_X;
    f->h = nr_X;
    f->w = nc_X;
    f->groups = num_group;
    f->n = nch_Y;
    f->binary = binary;
    f->xnor = xnor;
    f->batch = mini_batch_size;
    f->stride = stride;
    f->size = k_size;
    f->pad = padding;
    // K shape: nch_Y * group_size_X * nr_K * nc_K;
    f->nweights = nch_Y * nch_X / num_group * k_size * k_size;

    int nr_Y = math21_ml_function_conv_cal_nr_or_nc_Y(f->h, f->pad, f->size, f->stride);
    int nc_Y = math21_ml_function_conv_cal_nr_or_nc_Y(f->w, f->pad, f->size, f->stride);
    f->out_c = nch_Y;
    f->out_h = nr_Y;
    f->out_w = nc_Y;
    f->outputs = f->out_h * f->out_w * f->out_c;
    // X shape: mbs * num_group * group_size_X * nr_X * nc_X
    f->inputs = f->w * f->h * f->c;

    float scale = sqrt(2. / (k_size * k_size * nch_X / f->groups));
    int i;
#if defined(MATH21_FLAG_USE_CPU)
    f->weights = math21_vector_create_with_default_value_wrapper(f->nweights, 0);
    for (i = 0; i < f->nweights; ++i) f->weights[i] = scale * math21_pr_rand_normal();
#else
    f->weights_cpu = math21_vector_create_with_default_value_cpu(f->nweights, 0);
    for (i = 0; i < f->nweights; ++i) f->weights_cpu[i] = scale * math21_pr_rand_normal();
    f->weights = math21_vector_create_from_cpuvector_wrapper(f->nweights, f->weights_cpu, 1); // by cl
#endif
    f->weight_updates = math21_vector_create_with_default_value_wrapper(f->nweights, 0);
    f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);

    f_detail->K_wrapper.setWrapper(f->weights, f->out_c, f->c / f->groups, f->size, f->size);
    f_detail->dK_wrapper.setWrapper(f->weight_updates, f->out_c, f->c / f->groups, f->size, f->size);
    f_detail->y_wrapper.setWrapper(f->output, f->batch, f->out_c, f->out_h, f->out_w);
    f_detail->dy_wrapper.setWrapper(f->delta, f->batch, f->out_c, f->out_h, f->out_w);

    if (is_batch_normalize) {
        mlfunction_node finput = {0};
        finput.y = f->output;
        finput.dy = f->delta;
        f->bn = math21_ml_function_batchnorm_create(0, 0, &finput, mini_batch_size, nc_Y, nr_Y, nch_Y, adam);
    } else {
#if !defined(MATH21_FLAG_USE_CPU)
        f->biases_cpu = math21_vector_create_with_default_value_cpu(nch_Y, 0);
#endif
        f->biases = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
        f->bias_updates = math21_vector_create_with_default_value_wrapper(nch_Y, 0);

        f_detail->b_wrapper.setWrapper(f->biases, f->out_c);
        f_detail->db_wrapper.setWrapper(f->bias_updates, f->out_c);
    }

    if (adam) {
        f->m = math21_vector_create_with_default_value_wrapper(f->nweights, 0);
        f->v = math21_vector_create_with_default_value_wrapper(f->nweights, 0);
        if (!f->bn) {
            f->bias_m = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
            f->bias_v = math21_vector_create_with_default_value_wrapper(nch_Y, 0);
        }
    }

    if (binary) {
        math21_tool_assert(0);
#if defined(MATH21_FLAG_USE_CPU)
        f->binary_weights = math21_vector_create_from_cpuvector_wrapper(f->nweights, f->weights, 1);
#else
        f->binary_weights = math21_vector_create_from_cpuvector_wrapper(f->nweights, f->weights_cpu, 1);
#endif
    }

    if (xnor) {
        math21_tool_assert(0);
#if defined(MATH21_FLAG_USE_CPU)
        f->binary_weights = math21_vector_create_from_cpuvector_wrapper(f->nweights, f->weights, 1);
#else
        f->binary_weights = math21_vector_create_from_cpuvector_wrapper(f->nweights, f->weights_cpu, 1);
#endif
        f->binary_input = math21_vector_create_with_default_value_wrapper(f->inputs * f->batch, 0);
    }

    // X_prime shape (group_size_X * k_size * k_size ) * (nr_Y * nc_Y)
    f->workspace_size = _math21_ml_function_conv_get_X_prime_size(f->out_h, f->out_w,
                                                                  f->size, f->c, f->groups);
    f->activation = activation;

    f->name = math21_string_create_from_string("conv2d");
    if (fnode) {
        fnode->type = mlfnode_type_conv;
        fnode->function = f;
        fnode->saveState = math21_ml_function_conv_node_saveState;
        fnode->getGlobalSpaceSize = math21_ml_function_conv_node_getGlobalSpaceSize;
        fnode->set_mbs = math21_ml_function_conv_node_set_mbs;
        fnode->forward = math21_ml_function_conv_node_forward;
        fnode->backward = math21_ml_function_conv_node_backward;
        fnode->update = math21_ml_function_conv_node_update;
        fnode->log = math21_ml_function_conv_node_log;
        fnode->getDataToCpu = math21_ml_function_conv_node_getDataToCpu;
        fnode->getRawTensorToCpu = math21_ml_function_conv_node_getRawTensorToCpu;
        fnode->getName = math21_ml_function_conv_node_getName;
        math21_ml_function_conv_node_reset(fnode);
        f->fnode = fnode;
    }
    return f;
}

void math21_ml_function_conv_resize(mlfunction_node *fnode, mlfunction_conv *f, int nc_X, int nr_X) {
    auto *f_detail = new mlfunction_conv_detail();
    f->w = nc_X;
    f->h = nr_X;
    int nr_Y = math21_ml_function_conv_cal_nr_or_nc_Y(f->h, f->pad, f->size, f->stride);
    int nc_Y = math21_ml_function_conv_cal_nr_or_nc_Y(f->w, f->pad, f->size, f->stride);

    f->out_h = nr_Y;
    f->out_w = nc_Y;

    f->outputs = f->out_h * f->out_w * f->out_c;
    f->inputs = f->w * f->h * f->c;

    f->output = math21_vector_resize_with_default_value_wrapper(f->output, f->batch * f->outputs, 0);
    f->delta = math21_vector_resize_with_default_value_wrapper(f->delta, f->batch * f->outputs, 0);
    f_detail->y_wrapper.setWrapper(f->output, f->batch, f->out_c, f->out_h, f->out_w);
    f_detail->dy_wrapper.setWrapper(f->delta, f->batch, f->out_c, f->out_h, f->out_w);

    if (f->bn) {
        mlfunction_node finput = {0};
        finput.y = f->output;
        finput.dy = f->delta;
        math21_ml_function_batchnorm_resize(f->bn, &finput, nc_Y, nr_Y);
    }
    f->workspace_size = _math21_ml_function_conv_get_X_prime_size(f->out_h, f->out_w,
                                                                  f->size, f->c, f->groups);
    if (fnode) {
        math21_ml_function_conv_node_reset(fnode);
    }
}

int math21_ml_function_conv_cal_nr_or_nc_Y(int nr_X, int pad, int k_size, int stride) {
    return (nr_X + 2 * pad - k_size) / stride + 1;
}

void
math21_ml_function_conv_bias_backward(PointerFloatWrapper db, PointerFloatInputWrapper dY, int mini_batch_size,
                                      int features_size,
                                      int in_class_size) {
    math21_vector_sum_with_in_class_wrapper(db, dY, mini_batch_size, features_size, in_class_size);
}

void math21_ml_function_conv_swap_binary_weights(mlfunction_conv *f) {
    PointerFloatWrapper swap = f->weights;
    f->weights = f->binary_weights;
    f->binary_weights = swap;
}

//  Z = h(Y), Y = W*X + b, or Y = X*W.t + b
//  Y_m = K_m * X_prime + b ...
// float *workspace;// X_prime or dL/dX_prime
// workspace is global space, and has size at least workspace_size.
void math21_ml_function_conv_forward(mlfunction_conv *f, const mlfunction_node *finput0,
                                     int is_train, PointerFloatWrapper workspace) {
    if (is_train) {
        math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
    }
    mlfunction_node finput1 = *finput0;
    mlfunction_node *finput = &finput1;
    int imb, igroup;

    // Y_m = 0
    math21_vector_set_wrapper(f->outputs * f->batch, 0, f->output, 1);
    if (f->binary) {
        math21_ml_function_conv_binarize_weights_wrapper(f->weights, f->n, f->c / f->groups * f->size * f->size,
                                                         f->binary_weights);
        math21_ml_function_conv_swap_binary_weights(f);
    }

    if (f->xnor) {
        // K shape: nch_Y * group_size_X * nr_K * nc_K;
        math21_ml_function_conv_binarize_weights_wrapper(f->weights, f->n, f->c / f->groups * f->size * f->size,
                                                         f->binary_weights);
        math21_ml_function_conv_swap_binary_weights(f);
        math21_ml_function_conv_binarize_input_wrapper(finput->y, f->batch * f->c * f->h * f->w, f->binary_input);
        finput->y = f->binary_input;
    }

    // nr_Y_m = group_size_Y
    int nr_Y_m = f->n / f->groups;
    // n_common = group_size_X * nr_K * nc_K
    int n_common = (f->c / f->groups) * f->size * f->size;
    int nc_Y_m = f->out_w * f->out_h;
    for (imb = 0; imb < f->batch; ++imb) {
        for (igroup = 0; igroup < f->groups; ++igroup) {
            // K shape: nch_Y * group_size_X * nr_K * nc_K
            // K shape: (num_group * group_size_Y) * group_size_X * nr_K * nc_K
            // K shape: num_group * group_size_Y * n_common
            // K shape: num_group * nr_Y_m * n_common
            PointerFloatWrapper K_m = f->weights + igroup * (f->nweights / f->groups);
            PointerFloatWrapper X_prime = workspace;
            // X shape: mbs * num_group * group_size_X * nr_X * nc_X
            PointerFloatWrapper X = finput->y + (imb * f->groups + igroup) * (f->c / f->groups) * f->h * f->w;
            // Y shape: mbs * nch_Y * nr_Y * nc_Y
            // Y shape: mbs * num_group * group_size_Y * nr_Y * nc_Y
            // Y shape: mbs * num_group * nr_Y_m * nr_Y * nc_Y
            // Y shape: mbs * num_group * nr_Y_m * nc_Y_m
            PointerFloatWrapper Y_m = f->output + (imb * f->groups + igroup) * nr_Y_m * nc_Y_m;

            if (f->size == 1) {
                X_prime = X;
            } else {
                // X -> X_prime
                // X_prime shape: (group_size_X * nr_K * nc_K ) * (nr_Y * nc_Y)
                // X_prime shape: (group_size_X * nr_K * nc_K ) * nc_Y_m
                // X_prime shape: n_common * nc_Y_m
                math21_ml_function_conv_X_to_X_prime_wrapper(X, f->c / f->groups, f->h, f->w, f->size, f->stride,
                                                             f->pad,
                                                             X_prime);
            }
            // Y_m = K_m * X_prime + Y_m, K_m: nr_Y_m*n_common, X_prime: n_common*nc_Y_m, Y_m: nr_Y_m*nc_Y_m
            math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 0, nr_Y_m, nc_Y_m, n_common, 1, K_m, n_common,
                                                                X_prime, nc_Y_m, 1, Y_m, nc_Y_m);
        }
    }

    if (f->bn) {
        // ...
        mlfunction_batchnorm *fbn = f->bn;
        fbn->is_train = is_train;
        math21_ml_function_batchnorm_forward(fbn, 0);
    } else {
        // Y += b
        math21_vector_x_add_b_with_in_class_wrapper(f->output, f->biases, f->batch, f->n, f->out_h * f->out_w);
    }

    // Z = h(Y)
    math21_function_activation_vector_wrapper(f->output, f->outputs * f->batch, f->activation);
    if (f->binary || f->xnor) math21_ml_function_conv_swap_binary_weights(f);

    if (f->fnode && f->fnode->id == 106) {
//        math21_ml_function_conv_log(f, "*/K");
//        m21log("...........\n");
//        math21_ml_function_conv_log(f, "*/K");
//        auto data = (const NumR32 *)f->fnode->getDataToCpu(f->fnode, "*/K");
//        math21_tensor_4d_float_log_cpu("K", data, 18, 256, 1, 1);
    }
}

// dL/dZ => dL/dK, dL/dX
void math21_ml_function_conv_backward(mlfunction_conv *f, mlfunction_node *finput,
                                      int is_train, PointerFloatWrapper workspace) {
    int imb, igroup;
    int nr_dK_m = f->n / f->groups;
    int nc_dK_m = f->size * f->size * f->c / f->groups;
    int nc_dY_m = f->out_w * f->out_h;

#ifndef MATH21_FLAG_USE_CPU
    if (f->smooth) {
        math21_tool_assert(0);
        math21_ml_function_conv_smooth_wrapper(f, 5, f->smooth);
    }
#endif

    // dL/dY = dL/dZ *.ele h.d(Y)
    math21_function_activation_gradient_vector_wrapper(f->output, f->outputs * f->batch, f->activation, f->delta);

    if (f->bn) {
        mlfunction_batchnorm *fbn = f->bn;
        fbn->is_train = is_train;
        math21_ml_function_batchnorm_backward(fbn, 0);
    } else {
        // dL/db += sum(dL/dY(i))
        math21_ml_function_conv_bias_backward(f->bias_updates, f->delta, f->batch, f->n, nc_dY_m);
    }

    PointerFloatWrapper original_input = finput->y;
    if (f->xnor) finput->y = f->binary_input;

    for (imb = 0; imb < f->batch; ++imb) {
        for (igroup = 0; igroup < f->groups; ++igroup) {
            // dK shape: nch_Y * group_size_X * nr_K * nc_K
            // dK shape: (num_group * group_size_Y) * group_size_X * nr_K * nc_K
            // dK shape: num_group * group_size_Y * n_common
            // dK shape: num_group * nr_dY_m * n_common
            // dK shape: num_group * nr_dK_m * n_common
            PointerFloatWrapper dK_m = f->weight_updates + igroup * (f->nweights / f->groups);
            PointerFloatWrapper X_prime = workspace;
            // X shape: mbs * num_group * group_size_X * nr_X * nc_X
            PointerFloatWrapper X = finput->y + (imb * f->groups + igroup) * (f->c / f->groups) * f->h * f->w;
            // dY shape: mbs * nch_Y * nr_Y * nc_Y
            // dY shape: mbs * num_group * group_size_Y * nr_Y * nc_Y
            // dY shape: mbs * num_group * nr_dK_m * nr_Y * nc_Y
            // dY shape: mbs * num_group * nr_dK_m * nc_dY_m
            // dY shape: mbs * num_group * nr_dY_m * nc_dY_m
            PointerFloatWrapper dY_m = f->delta + (imb * f->groups + igroup) * nr_dK_m * nc_dY_m;
            // if error, remove this, because stride>1 leads to different X_prime.
            if (f->size == 1) {
                X_prime = X;
            } else {
                // X -> X_prime
                math21_ml_function_conv_X_to_X_prime_wrapper(X, f->c / f->groups, f->h, f->w,
                                                             f->size, f->stride, f->pad, X_prime);
            }
            // dL/dW += dL/dY * X.t
            // dL/dK_m += dL/dY_m * X_prime.t
            math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 1, nr_dK_m, nc_dK_m, nc_dY_m, 1, dY_m, nc_dY_m,
                                                                X_prime, nc_dY_m, 1, dK_m, nc_dK_m);

            if (!math21_vector_isEmpty_wrapper(finput->dy)) {
                if (f->binary || f->xnor) math21_ml_function_conv_swap_binary_weights(f);
                PointerFloatWrapper K_m = f->weights + igroup * f->nweights / f->groups;
                PointerFloatWrapper dX_prime = workspace;
                PointerFloatWrapper dX = finput->dy + (imb * f->groups + igroup) * (f->c / f->groups) * f->h * f->w;
                // if error, remove this
                if (f->size == 1) {
                    dX_prime = dX;
                }

                // dL/dX = W.t * dL/dY
                // dL/dX_prime = K_m.t * dL/dY_m
                math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(1, 0, nc_dK_m, nc_dY_m, nr_dK_m, 1, K_m, nc_dK_m,
                                                                    dY_m, nc_dY_m, 0, dX_prime, nc_dY_m);

                if (f->size != 1) {
                    // dX_prime -> dX
                    math21_ml_function_conv_dX_prime_to_dX_wrapper(dX_prime, f->c / f->groups, f->h, f->w, f->size,
                                                                   f->stride, f->pad, dX);
                }
                if (f->binary || f->xnor) {
                    math21_ml_function_conv_swap_binary_weights(f);
                }
            }
            if (f->xnor) {
                math21_function_activation_gradient_vector_wrapper(original_input + imb * f->c * f->h * f->w,
                                                                   f->c * f->h * f->w,
                                                                   MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN,
                                                                   finput->dy + imb * f->c * f->h * f->w);
            }
        }
    }
}

void math21_ml_function_conv_update(mlfunction_conv *f, OptUpdate *optUpdate) {
    OptUpdate_Adam *a = 0;
    if (optUpdate->type == OptUpdateType_Adam) {
        a = (OptUpdate_Adam *) optUpdate->detail;
    }
    float learning_rate = optUpdate->alpha * f->learning_rate_scale;
    float momentum = optUpdate->momentum;
    float decay = optUpdate->decay;
    int batch = optUpdate->mini_batch_size;

    if (a) {
        math21_optimization_adam_update_wrapper(f->weights, f->weight_updates, f->m, f->v, a->beta1, a->beta2,
                                                a->eps, decay, learning_rate, f->nweights, batch, a->t);
        if (f->bn) {
            f->bn->learning_rate_scale = f->learning_rate_scale;
            math21_ml_function_batchnorm_update(f->bn, optUpdate);
        } else {
            math21_optimization_adam_update_wrapper(f->biases, f->bias_updates, f->bias_m, f->bias_v, a->beta1,
                                                    a->beta2, a->eps, decay, learning_rate, f->n, batch, a->t);
        }
    } else {
        if (f->bn) {
            f->bn->learning_rate_scale = f->learning_rate_scale;
            math21_ml_function_batchnorm_update(f->bn, optUpdate);
        } else {
            // b = b - alpha * dL/db
            // f->bias_updates = -dL/db because of loss function L is -L.
            math21_vector_kx_add_y_wrapper(f->n, learning_rate / batch, f->bias_updates, 1, f->biases, 1);
            // dL/db = momentum * dL/db
            math21_vector_kx_wrapper(f->n, momentum, f->bias_updates, 1);
        }

        // dL/dW = dL/dW + decay * W
        math21_vector_kx_add_y_wrapper(f->nweights, -decay * batch, f->weights, 1, f->weight_updates, 1);
        // W = W - alpha * dL/dW
        math21_vector_kx_add_y_wrapper(f->nweights, learning_rate / batch, f->weight_updates, 1, f->weights, 1);
        // dL/dW = momentum * dL/dW
        math21_vector_kx_wrapper(f->nweights, momentum, f->weight_updates, 1);
    }
    if (f->clip) {
        math21_tool_assert(0);
        math21_vector_clip_wrapper(f->nweights, f->clip, f->weights, 1);
    }
}

void math21_ml_function_conv_saveState(const mlfunction_conv *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->weights, f->nweights);
    math21_vector_serialize_c_wrapper(file, f->weight_updates, f->nweights);
    if (f->bn) {
        math21_ml_function_batchnorm_saveState(f->bn, file);
    } else {
        math21_vector_serialize_c_wrapper(file, f->biases, f->out_c);
        math21_vector_serialize_c_wrapper(file, f->bias_updates, f->out_c);
    }
}

// merge f to fb
void math21_ml_function_conv_merge_to(mlfunction_conv *f, mlfunction_conv *fb) {
    math21_vector_kx_add_y_cpu(f->nweights, 1, f->weights_cpu, 1, fb->weights_cpu, 1);
    if (f->bn) {
        math21_ml_function_batchnorm_merge_to(f->bn, fb->bn);
    } else {
        math21_vector_kx_add_y_cpu(f->n, 1, f->biases_cpu, 1, fb->biases_cpu, 1);
    }
}

void math21_ml_function_conv_scale(mlfunction_conv *f, float s) {
    math21_vector_kx_cpu(f->nweights, s, f->weights_cpu, 1);
    if (f->bn) {
        math21_ml_function_batchnorm_scale(f->bn, s);
    } else {
        math21_vector_kx_cpu(f->n, s, f->biases_cpu, 1);
    }
}

void math21_ml_function_conv_pull_wrapper(mlfunction_conv *f, NumB useRolling) {
    math21_vector_pull_wrapper(f->weights, f->weights_cpu, f->nweights);
    // ye
//    math21_cuda_pull_array(f->weight_updates, f->weight_updates, f->nweights);
//    math21_cuda_pull_array(f->bias_updates, f->bias_updates, f->n);
    if (f->bn) {
        math21_ml_function_batchnorm_pull_wrapper(f->bn, useRolling);
    } else {
        math21_vector_pull_wrapper(f->biases, f->biases_cpu, f->n);
    }
}

void math21_ml_function_conv_push_wrapper(mlfunction_conv *f, NumB useRolling) {
    math21_ml_function_conv_push_by_wrapper(f, f, useRolling);
}

// f is pushed by fb
void math21_ml_function_conv_push_by_wrapper(mlfunction_conv *f, mlfunction_conv *fb, NumB useRolling) {
    math21_vector_push_wrapper(f->weights, fb->weights_cpu, f->nweights);
    // by ye
    // math21_vector_push_wrapper(f->weight_updates, fb->weight_updates, f->nweights);
    // math21_vector_push_wrapper(f->bias_updates, fb->bias_updates, f->n);
    if (f->bn) {
        math21_ml_function_batchnorm_push_by_wrapper(f->bn, fb->bn, useRolling);
    } else {
        math21_vector_push_wrapper(f->biases, fb->biases_cpu, f->n);
    }
}

void math21_ml_function_conv_set_mbs(mlfunction_conv *f, int mini_batch_size) {
    f->batch = mini_batch_size;
    if (f->bn) {
        f->bn->mini_batch_size = mini_batch_size;
    }
}
