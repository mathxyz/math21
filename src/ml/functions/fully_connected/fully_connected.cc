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

#include "../conv/files_c.h"
#include "inner_cc.h"
#include "fully_connected.h"

using namespace math21;

void math21_ml_function_fully_connected_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                              const mlfunction_node *finput, m21list *options) {
    int output = math21_function_option_find_int(options, "output", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "logistic");
    const char *name = math21_function_option_find_str_quiet(options, "name", 0);
    MATH21_FUNCTION_ACTIVATION_TYPE activation = math21_function_activation_get_type(activation_s);
    int is_use_bias = math21_function_option_find_int_quiet(options, "is_use_bias", 1);
    int is_batch_normalize = math21_function_option_find_int_quiet(options, "batch_normalize", 0);

    mlfunction_fully_connected *f = math21_ml_function_fully_connected_create(fnode, finput->mini_batch_size,
                                                                              finput->y_size, output,
                                                                              activation, is_use_bias,
                                                                              is_batch_normalize, fnet->adam, name);
    f->learning_rate_scale = math21_function_option_find_float_quiet(options, "learning_rate", 1);
}

// merge f to fb
void math21_ml_function_fully_connected_merge_to(mlfunction_fully_connected *f, mlfunction_fully_connected *fb) {
    math21_vector_kx_add_y_cpu(f->nweights, 1, f->weights_cpu, 1, fb->weights_cpu, 1);
    if (f->bn) {
        math21_ml_function_batchnorm_merge_to(f->bn, fb->bn);
    } else {
        if (f->is_use_bias) {
            math21_vector_kx_add_y_cpu(f->outputs, 1, f->biases_cpu, 1, fb->biases_cpu, 1);
        }
    }
}

void math21_ml_function_fully_connected_scale(mlfunction_fully_connected *f, float s) {
    math21_vector_kx_cpu(f->nweights, s, f->weights_cpu, 1);
    if (f->bn) {
        math21_ml_function_batchnorm_scale(f->bn, s);
    } else {
        if (f->is_use_bias) {
            math21_vector_kx_cpu(f->outputs, s, f->biases_cpu, 1);
        }
    }
}

void math21_ml_function_fully_connected_pull_wrapper(mlfunction_fully_connected *f, NumB useRolling) {
    math21_vector_pull_wrapper(f->weights, f->weights_cpu, f->nweights);
    if (f->bn) {
        math21_ml_function_batchnorm_pull_wrapper(f->bn, useRolling);
    } else {
        if (f->is_use_bias) {
            math21_vector_pull_wrapper(f->biases, f->biases_cpu, f->outputs);
        }
    }
}

void math21_ml_function_fully_connected_push_wrapper(mlfunction_fully_connected *f, NumB useRolling) {
    math21_ml_function_fully_connected_push_by_wrapper(f, f, useRolling);
}

// f is pushed by fb
void math21_ml_function_fully_connected_push_by_wrapper(mlfunction_fully_connected *f, mlfunction_fully_connected *fb, NumB useRolling) {
    math21_vector_push_wrapper(f->weights, fb->weights_cpu, f->nweights);
    if (f->bn) {
        math21_ml_function_batchnorm_push_by_wrapper(f->bn, fb->bn, useRolling);
    } else {
        if (f->is_use_bias) {
            math21_vector_push_wrapper(f->biases, fb->biases_cpu, f->outputs);
        }
    }
}

void math21_ml_function_fully_connected_save_theta_order_bwsmv(mlfunction_fully_connected *f, FILE *fp) {
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_fully_connected_pull_wrapper(f, 1);
#endif

    float *weights = 0;
    float *biases = 0;
    float *scales = 0;
    float *rolling_mean = 0;
    float *rolling_variance = 0;
#if defined(MATH21_FLAG_USE_CPU)
    weights = f->weights;
    if (f->bn) {
        biases = f->bn->biases;
        scales = f->bn->scales;
        rolling_mean = f->bn->rolling_mean;
        rolling_variance = f->bn->rolling_variance;
    } else {
        biases = f->biases;
    }
#else
    weights = f->weights_cpu;
    if (f->bn) {
        biases = f->bn->biases_cpu;
        scales = f->bn->scales_cpu;
        rolling_mean = f->bn->rolling_mean_cpu;
        rolling_variance = f->bn->rolling_variance_cpu;
    } else {
        biases = f->biases_cpu;
    }
#endif

    int num = f->nweights;
    if (biases) {
        fwrite(biases, sizeof(float), f->outputs, fp);
    }
    fwrite(weights, sizeof(float), num, fp);
    if (scales)fwrite(scales, sizeof(float), f->outputs, fp);
    if (rolling_mean)fwrite(rolling_mean, sizeof(float), f->outputs, fp);
    if (rolling_variance)fwrite(rolling_variance, sizeof(float), f->outputs, fp);
}

void math21_ml_function_fully_connected_load_theta_order_bwsmv(mlfunction_fully_connected *f, FILE *fp) {
    float *weights = 0;
    float *biases = 0;
    float *scales = 0;
    float *rolling_mean = 0;
    float *rolling_variance = 0;
#if defined(MATH21_FLAG_USE_CPU)
    weights = f->weights;
    if (f->bn) {
        biases = f->bn->biases;
        scales = f->bn->scales;
        rolling_mean = f->bn->rolling_mean;
        rolling_variance = f->bn->rolling_variance;
    } else {
        biases = f->biases;
    }
#else
    weights = f->weights_cpu;
    if (f->bn) {
        biases = f->bn->biases_cpu;
        scales = f->bn->scales_cpu;
        rolling_mean = f->bn->rolling_mean_cpu;
        rolling_variance = f->bn->rolling_variance_cpu;
    } else {
        biases = f->biases_cpu;
    }
#endif

    int num = f->nweights;
    if (biases) {
        fread(biases, sizeof(float), f->outputs, fp);
    }
    fread(weights, sizeof(float), num, fp);
    if (scales)fread(scales, sizeof(float), f->outputs, fp);
    if (rolling_mean)fread(rolling_mean, sizeof(float), f->outputs, fp);
    if (rolling_variance)fread(rolling_variance, sizeof(float), f->outputs, fp);
    if (f->flipped) {
        math21_matrix_transpose(weights, f->inputs, f->outputs);
    }
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_fully_connected_push_wrapper(f, 1);
#endif
}

void math21_ml_function_fully_connected_load_theta_order_bwsmv_flipped(mlfunction_fully_connected *f, FILE *fp,
                                                                       int flipped) {
    f->flipped = flipped;
    math21_ml_function_fully_connected_load_theta_order_bwsmv(f, fp);
}

void math21_ml_function_fully_connected_save_theta(mlfunction_fully_connected *f, FILE *fp) {
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_fully_connected_pull_wrapper(f, 1);
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
        if (biases) {
            fwrite(biases, sizeof(float), f->outputs, fp);
        }
    }
    fwrite(weights, sizeof(float), num, fp);
}

void math21_ml_function_fully_connected_load_theta(mlfunction_fully_connected *f, FILE *fp) {
#if defined(MATH21_FLAG_USE_CPU)
    float * weights = f->weights;
    float * biases = f->biases;
#else
    float *weights = f->weights_cpu;
    float *biases = f->biases_cpu;
#endif

    int num = f->nweights;
    if (f->bn) {
        math21_ml_function_batchnorm_load_theta(f->bn, fp, 0);
    } else {
        if (biases) {
            fread(biases, sizeof(float), f->outputs, fp);
        }
    }
    fread(weights, sizeof(float), num, fp);
    if (f->flipped) {
        math21_matrix_transpose(weights, f->inputs, f->outputs);
    }
#ifndef MATH21_FLAG_USE_CPU
    math21_ml_function_fully_connected_push_wrapper(f, 1);
#endif
}

void math21_ml_function_fully_connected_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_saveState(f, file);
}

// todo: check relation with n_time_step;
void math21_ml_function_fully_connected_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_set_mbs(f, mini_batch_size);
}

void
math21_ml_function_fully_connected_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_forward(f, finput, net->is_train);
}

void
math21_ml_function_fully_connected_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_backward(f, finput, net->is_train);
}

void math21_ml_function_fully_connected_node_update(mlfunction_node *fnode, OptUpdate *optUpdate) {
    auto *f = (mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_update(f, optUpdate);
}

void math21_ml_function_fully_connected_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_fully_connected *) fnode->function;
    math21_ml_function_fully_connected_log(f, varName);
}

const char *math21_ml_function_fully_connected_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_fully_connected *) fnode->function;
    return f->name;
}

void math21_ml_function_fully_connected_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_fully_connected *) fnode->function;
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

void math21_ml_function_fully_connected_destroy(mlfunction_fully_connected *f) {
    // reset first
    math21_ml_function_fully_connected_reset(f);
    // then destroy
}

// y = W*x + b
void math21_ml_function_fully_connected_log(const mlfunction_fully_connected *f, const char *varName) {
    std::string _varNameNew;
    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
        return;
    }
    varName = _varNameNew.c_str();

    if (math21_string_is_equal(varName, "summary")) {
        if (f->n_time_step > 1) {
            fprintf(stdout, "%s with time: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d)\n", f->name,
                    f->h, f->w, f->c, f->n_time_step, f->batch, f->out_h, f->out_w, f->out_c, f->n_time_step, f->batch);
        } else {
            fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d)\n", f->name,
                    f->h, f->w, f->c, f->total_mbs, f->out_h, f->out_w, f->out_c, f->total_mbs);
        }
        return;
    }
    fprintf(stdout, "%s:\n", f->name);
    std::string name = varName;
    if (name == "y") {
        if (f->out_h == 1 && f->out_w == 1) {
            math21_tensor_2d_float_log_wrapper(varName, f->output, f->total_mbs, f->out_c);
        } else {
            math21_tensor_4d_float_log_wrapper(varName, f->output, f->total_mbs, f->out_c, f->out_h, f->out_w);
        }
    } else if (name == "dy") {
        if (f->out_h == 1 && f->out_w == 1) {
            math21_tensor_2d_float_log_wrapper(varName, f->delta, f->total_mbs, f->out_c);
        } else {
            math21_tensor_4d_float_log_wrapper(varName, f->delta, f->total_mbs, f->out_c, f->out_h, f->out_w);
        }
    } else if (name == "W") {
        math21_tensor_2d_float_log_wrapper(varName, f->weights, f->outputs, f->inputs);
    } else if (name == "dW") {
        math21_tensor_2d_float_log_wrapper(varName, f->weight_updates, f->outputs, f->inputs);
    } else if (name == "b") {
        if (f->bn) {

        } else {
            if (f->is_use_bias) {
                math21_tensor_1d_float_log_wrapper(varName, f->biases, f->outputs);
            }
        }
    } else if (name == "db") {
        if (f->bn) {

        } else {
            if (f->is_use_bias) {
                math21_tensor_1d_float_log_wrapper(varName, f->bias_updates, f->outputs);
            }
        }
    } else {
        m21log("no variable name ", varName);
    }
}

// Z = h(Y), Y = X*W.t + b
mlfunction_fully_connected *math21_ml_function_fully_connected_create(
        mlfunction_node *fnode, int batch_size, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
        const char *name) {
    auto *f = (mlfunction_fully_connected *) math21_vector_calloc_cpu(1, sizeof(mlfunction_fully_connected));
    f->learning_rate_scale = 1;

    f->inputs = input_size;
    f->outputs = output_size;
    f->batch = batch_size;
    f->h = 1;
    f->w = 1;
    f->c = input_size;
    f->out_h = 1;
    f->out_w = 1;
    f->out_c = output_size;

    f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);

    int nweights = output_size * input_size;
    f->nweights = nweights;
    float scale = sqrt(2. / input_size);
    int i;
#if defined(MATH21_FLAG_USE_CPU)
    f->weights = math21_vector_create_with_default_value_wrapper(nweights, 0);
    for (i = 0; i < nweights; ++i) f->weights[i] = scale * math21_pr_rand_uniform(-1, 1);
#else
    f->weights_cpu = math21_vector_create_with_default_value_cpu(nweights, 0);
//    for (i = 0; i < nweights; ++i) f->weights_cpu[i] = scale * math21_pr_rand_uniform(-1, 1);
    if (math21_ml_function_tool_is_debug()) {
        for (i = 0; i < nweights; ++i) f->weights_cpu[i] = 1;
    } else {
        for (i = 0; i < nweights; ++i) f->weights_cpu[i] = scale * math21_pr_rand_uniform(-1, 1);
    }
    f->weights = math21_vector_create_from_cpuvector_wrapper(nweights, f->weights_cpu, 1);
#endif
    f->weight_updates = math21_vector_create_with_default_value_wrapper(nweights, 0);

    if (is_batch_normalize) {
        mlfunction_node finput = {0};
        finput.y = f->output;
        finput.dy = f->delta;
        f->bn = math21_ml_function_batchnorm_create(0, 0, &finput, batch_size, f->out_w, f->out_h, f->out_c, is_adam);
    } else {
        if (is_use_bias) {
            f->is_use_bias = is_use_bias;
#ifndef MATH21_FLAG_USE_CPU
            f->biases_cpu = math21_vector_create_with_default_value_cpu(output_size, 0);
#endif
            f->biases = math21_vector_create_with_default_value_wrapper(output_size, 0);
            f->bias_updates = math21_vector_create_with_default_value_wrapper(output_size, 0);
        } else {
            // f->biases is empty already.
        }
    }

    if (is_adam) {
        f->m = math21_vector_create_with_default_value_wrapper(nweights, 0);
        f->v = math21_vector_create_with_default_value_wrapper(nweights, 0);
        if (!f->bn) {
            if (f->is_use_bias) {
                f->bias_m = math21_vector_create_with_default_value_wrapper(f->outputs, 0);
                f->bias_v = math21_vector_create_with_default_value_wrapper(f->outputs, 0);
            }
        }
    }

    f->activation = activation;
    f->total_mbs = f->batch;
    f->n_time_step = 1;
    f->i_time_step = 0;
    if (!name) {
        name = "fully connected";
    }
    f->name = math21_string_create_from_string(name);
    if (fnode) {
        fnode->type = mlfnode_type_fully_connected;
        fnode->function = f;
        fnode->saveState = math21_ml_function_fully_connected_node_saveState;
        fnode->set_mbs = math21_ml_function_fully_connected_node_set_mbs;
        fnode->forward = math21_ml_function_fully_connected_node_forward;
        fnode->backward = math21_ml_function_fully_connected_node_backward;
        fnode->update = math21_ml_function_fully_connected_node_update;
        fnode->log = math21_ml_function_fully_connected_node_log;
        fnode->getName = math21_ml_function_fully_connected_node_getName;
        math21_ml_function_fully_connected_node_reset(fnode);
    }
    return f;
}

mlfunction_fully_connected *math21_ml_function_fully_connected_with_n_time_step_create(
        mlfunction_node *fnode, int rnn_batch_size, int n_time_step, int input_size, int output_size,
        MATH21_FUNCTION_ACTIVATION_TYPE activation, int is_use_bias, int is_batch_normalize, int is_adam,
        const char *name) {
    math21_tool_assert(n_time_step > 0);
    mlfunction_fully_connected *f = math21_ml_function_fully_connected_create(fnode, rnn_batch_size * n_time_step,
                                                                              input_size,
                                                                              output_size, activation,
                                                                              is_use_bias,
                                                                              is_batch_normalize,
                                                                              is_adam, name);
    f->n_time_step = n_time_step;
    f->batch = rnn_batch_size;
    if (f->bn) {
        f->bn->n_time_step = n_time_step;
        f->bn->mini_batch_size = rnn_batch_size;
    }
    return f;
}

// Z = h(Y), Y = W*X + b, or Y = X*W.t + b
void math21_ml_function_fully_connected_forward(
        mlfunction_fully_connected *f, mlfunction_node *finput, int is_train) {
    if (is_train) {
        if (f->i_time_step == 0) {
            math21_vector_set_wrapper(f->total_mbs * f->outputs, 0, f->delta, 1);
        }
    }
    math21_vector_set_wrapper(f->outputs * f->batch, 0, f->output, 1);
    int sb = f->batch;
    int sx = f->inputs;
    int sy = f->outputs;
    PointerFloatWrapper x = finput->y;
    PointerFloatWrapper w = f->weights;
    PointerFloatWrapper y = f->output;

    // Y += X*W.t
    math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 1, sb, sy, sx, 1, x, sx, w, sx, 1, y, sy);

    if (f->bn) {
        // Y = BN(Y)
        mlfunction_batchnorm *fbn = f->bn;
        fbn->is_train = is_train;
        math21_ml_function_batchnorm_forward(fbn, 0);
    } else {
        // Y += b
        if (f->is_use_bias) {
            math21_vector_x_add_b_with_in_class_wrapper(f->output, f->biases, f->batch, f->outputs, 1);
        }
    }

    // Z = h(Y)
    math21_function_activation_vector_wrapper(f->output, f->outputs * f->batch, f->activation);
}

// Z = h(Y), Y = W*X + b, or Y = X*W.t + b
// dL/dZ => dL/dW, dL/dX
void math21_ml_function_fully_connected_backward(
        mlfunction_fully_connected *f, mlfunction_node *finput, int is_train) {
    if (f->bn) {
        math21_vector_clip_wrapper(f->outputs * f->batch, 1, f->delta, 1);
    }

    // dL/dY = dL/dZ *.ele h.d(Y)
    math21_function_activation_gradient_vector_wrapper(f->output, f->outputs * f->batch, f->activation, f->delta);

    if (f->bn) {
        mlfunction_batchnorm *fbn = f->bn;
        fbn->is_train = is_train;
        math21_ml_function_batchnorm_backward(fbn, 0);
    } else {
        if (f->is_use_bias) {
            // dL/db += sum(dL/dY(i))
            math21_ml_function_conv_bias_backward(f->bias_updates, f->delta, f->batch, f->outputs, 1);
        }
    }

    int m = f->outputs;
    int k = f->batch;
    int n = f->inputs;
    PointerFloatWrapper a = f->delta;
    PointerFloatWrapper b = finput->y;
    PointerFloatWrapper c = f->weight_updates;
    // dL/dW += dL/dY * X.t
    math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = f->batch;
    k = f->outputs;
    n = f->inputs;

    a = f->delta;
    b = f->weights;
    c = finput->dy;

    // dL/dX = W.t * dL/dY
    // but here is addTo dX ...
    if (!math21_vector_isEmpty_wrapper(finput->dy)) {
        math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
}

void math21_ml_function_fully_connected_update(mlfunction_fully_connected *f, OptUpdate *optUpdate) {
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
                                                a->eps, decay, learning_rate, f->inputs * f->outputs, batch, a->t);
        if (f->bn) {
            f->bn->learning_rate_scale = f->learning_rate_scale;
            math21_ml_function_batchnorm_update(f->bn, optUpdate);
        } else {
            if (f->is_use_bias) {
                math21_optimization_adam_update_wrapper(f->biases, f->bias_updates, f->bias_m, f->bias_v, a->beta1,
                                                        a->beta2, a->eps, decay, learning_rate, f->outputs, batch,
                                                        a->t);
            }
        }
    } else {
        if (f->bn) {
            f->bn->learning_rate_scale = f->learning_rate_scale;
            math21_ml_function_batchnorm_update(f->bn, optUpdate);
        } else {
            if (f->is_use_bias) {
                // b = b - alpha * dL/db
                // f->bias_updates = -dL/db because of loss function L is -L.
                math21_vector_kx_add_y_wrapper(f->outputs, learning_rate / batch, f->bias_updates, 1, f->biases, 1);
                // dL/db = momentum * dL/db
                math21_vector_kx_wrapper(f->outputs, momentum, f->bias_updates, 1);
            }
        }

        math21_vector_kx_add_y_wrapper(f->inputs * f->outputs, -decay * batch, f->weights, 1, f->weight_updates, 1);
        math21_vector_kx_add_y_wrapper(f->inputs * f->outputs, learning_rate / batch, f->weight_updates, 1,
                                       f->weights, 1);
        math21_vector_kx_wrapper(f->inputs * f->outputs, momentum, f->weight_updates, 1);
    }
}

void math21_ml_function_fully_connected_saveState(const mlfunction_fully_connected *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->total_mbs * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->total_mbs * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->weights, f->nweights);
    math21_vector_serialize_c_wrapper(file, f->weight_updates, f->nweights);
    if (f->bn) {
        math21_ml_function_batchnorm_saveState(f->bn, file);
    } else {
        if (f->is_use_bias) {
            math21_vector_serialize_c_wrapper(file, f->biases, f->outputs);
            math21_vector_serialize_c_wrapper(file, f->bias_updates, f->outputs);
        }
    }
}

void math21_ml_function_fully_connected_increase_by_time(mlfunction_fully_connected *f, int time_steps) {
    f->i_time_step += time_steps;

    int num = f->outputs * f->batch * time_steps;
    f->output += num;
    f->delta += num;
    if (f->bn) {
        math21_ml_function_batchnorm_increase_by_time(f->bn, time_steps);
    }
}

void math21_ml_function_fully_connected_reset(mlfunction_fully_connected *f) {
    int num = f->outputs * f->batch * f->i_time_step;
    f->output -= num;
    f->delta -= num;
    f->i_time_step = 0;

    if (f->bn) {
        math21_ml_function_batchnorm_reset(f->bn);
    }
}

void math21_ml_function_fully_connected_set_mbs(mlfunction_fully_connected *f, int mini_batch_size) {
    f->batch = mini_batch_size;
    if (f->bn) {
        f->bn->mini_batch_size = mini_batch_size;
    }
}
