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

#include "../../../image/files_c.h"
#include "../tool/files_c.h"
#include "inner_cc.h"
#include "net.h"
#include "detail.h"

using namespace math21;

mlfunction_net *math21_ml_function_net_create(int n_node) {
    mlfunction_net *net = (mlfunction_net *) math21_vector_calloc_cpu(1, sizeof(mlfunction_net));
    net->n_node = n_node;
    net->nodes = (mlfunction_node **) math21_vector_calloc_cpu((size_t) net->n_node, sizeof(mlfunction_node *));
    int i;
    for (i = 0; i < net->n_node; ++i) {
        net->nodes[i] = math21_ml_function_node_create();
        net->nodes[i]->id = i + 1;
    }
    auto *f_detail = new mlfunction_net_detail();
    net->detail = f_detail;
    return net;
}

void math21_ml_function_net_destroy(mlfunction_net *net) {
    if (!net) {
        return;
    }
    int i;
    for (i = 0; i < net->n_node; ++i) {
        math21_ml_function_node_destroy(net->nodes[i]);
    }
    math21_vector_free_cpu(net->nodes);
    math21_vector_free_cpu(net);
}

mlfunction_node *math21_ml_function_net_get_output_node(mlfunction_net *net) {
    int i;
    for (i = net->n_node - 1; i >= 0; --i) {
        if (net->nodes[i]->type != mlfnode_type_cost) break;
    }
    return net->nodes[i];
}

void math21_ml_function_net_calculate_cost(mlfunction_net *fnet) {
    int i;
    float sum = 0;
    int count = 0;
    for (i = 0; i < fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->getCost) {
            sum += fnode->getCost(fnode);
            ++count;
        }
    }
    fnet->cost = sum / count;
}

#ifndef MATH21_FLAG_USE_CPU

// used by rnn
void math21_ml_function_net_pull_output_wrapper(mlfunction_net *fnet) {
    math21_vector_pull_wrapper(fnet->y_wrapper, fnet->y_cpu, fnet->mini_batch_size * fnet->y_size);
}

#endif

void math21_ml_function_net_forward(mlfunction_net *fnet) {
#ifndef MATH21_FLAG_USE_CPU
    math21_gpu_set_device_wrapper(fnet->gpuDevice);
    math21_vector_push_wrapper(fnet->data_x_wrapper, fnet->data_x_cpu, fnet->mini_batch_size * fnet->data_x_size);
    if (fnet->data_y_cpu) {
        math21_vector_push_wrapper(fnet->data_y_wrapper, fnet->data_y_cpu, fnet->mbs_y * fnet->data_y_size);
    }
#endif

    mlfunction_node finput0 = {0};
    finput0.mini_batch_size = fnet->mini_batch_size;
    finput0.y_size = fnet->data_x_size;
    finput0.y = fnet->data_x_wrapper;
    fnet->finput = &finput0;

    mlfunction_node ftruth0 = {0};
    ftruth0.mini_batch_size = fnet->mbs_y;
    ftruth0.y_size = fnet->data_y_size;
    ftruth0.y = fnet->data_y_wrapper;
    fnet->ftruth = &ftruth0;

    int i;
    for (i = 1; i <= fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i - 1];

        math21_time_index_add();
//        if(i==fnet->n_node){
//            printf("fw time end: %d\n", math21_time_index_get());
//        }

#ifdef MATH21_FLAG_USE_CPU
        if (math21_ml_function_net_getlogLevel() > 1) {
            printf("\rforward node %d", i);
            fflush(stdout);
        }
#endif
        fnode->forward(fnode, fnet, fnet->finput);
        fnet->finput = fnode;
        // todo: put it to fnode->function
//        if (fnode->truth) {
//            fnet->ftruth = fnode;
//        }
        math21_ml_function_debug_node_save_state(fnode);
    }
    math21_ml_function_net_calculate_cost(fnet);
#ifdef MATH21_FLAG_USE_CPU
    if (math21_ml_function_net_getlogLevel() > 1) {
        printf("\r                 \r");// carriage return
    }
#endif
}

void math21_ml_function_net_backward(mlfunction_net *fnet) {
    mlfunction_node finput0 = {0};
#ifdef MATH21_FLAG_USE_CPU
    finput0.y = fnet->data_x_wrapper;
#else
    math21_gpu_set_device_wrapper(fnet->gpuDevice);
    finput0.y = fnet->data_x_wrapper;
#endif
    int i;
    for (i = fnet->n_node; i >= 1; --i) {
#ifdef MATH21_FLAG_USE_CPU
        if (math21_ml_function_net_getlogLevel() > 1) {
            printf("\rbackward node %d", i);
            fflush(stdout);
        }
#endif
        mlfunction_node *fnode_pre = 0;
        if (i >= 2) {
            fnode_pre = fnet->nodes[i - 2];
        }
        mlfunction_node *fnode = fnet->nodes[i - 1];

        math21_time_index_add();

//        if(i==1){
//            printf("bw time end: %d\n", math21_time_index_get());
//        }

        if (fnode->stopbackward) break;
        if (i == 1) {
            fnet->finput = &finput0;
        } else {
            fnet->finput = fnode_pre;
        }

        fnode->backward(fnode, fnet, fnet->finput);

        math21_ml_function_debug_node_save_state(fnode);
    }
#ifdef MATH21_FLAG_USE_CPU
    if (math21_ml_function_net_getlogLevel() > 1) {
        printf("\r                  \r");
    }
#endif
}

// nmb >= 0, but often > 0
NumSize math21_ml_function_net_get_update_count(mlfunction_net *fnet) {
    NumSize nmb = fnet->n_seen / (fnet->mini_batch_size_in_opt);
    return nmb;
}

int math21_ml_function_net_should_train_continue(mlfunction_net *fnet) {
    return math21_ml_function_net_get_update_count(fnet) < fnet->n_mini_batch_max_in_opt ||
           fnet->n_mini_batch_max_in_opt == 0;
}

float math21_ml_function_net_opt_get_alpha(mlfunction_net *fnet) {
    m21OptAlphaPolicyConfig config;
    // set all elements.
    config.n_mini_batch_max_in_opt = fnet->n_mini_batch_max_in_opt;
    config.t = math21_ml_function_net_get_update_count(fnet);
    config.alphaPolicy = fnet->alphaPolicy;
    config.burn_in = fnet->burn_in;
    config.alpha = fnet->alpha;
    config.momentum = fnet->momentum;
    config.decay = fnet->decay;
    config.adam = fnet->adam;
    config.B1 = fnet->B1;
    config.B2 = fnet->B2;
    config.eps = fnet->eps;
    config.num_steps = fnet->num_steps;
    config.steps = fnet->steps;
    config.scales = fnet->scales;
    config.gamma = fnet->gamma;
    config.step = fnet->step;
    config.power = fnet->power;
    config.scale = fnet->scale;

    return math21_opt_get_alpha_by_policy(&config);
}

void math21_ml_function_net_opt_update(mlfunction_net *fnet) {
#ifndef MATH21_FLAG_USE_CPU
    math21_gpu_set_device_wrapper(fnet->gpuDevice);
#endif
    ++fnet->time_step_in_opt;

    OptUpdate optUpdate;
    math21_vector_memset_cpu(&optUpdate, 0, sizeof(OptUpdate));
    OptUpdate_Adam optUpdateAdam = {0};

    if (fnet->adam) {
        optUpdate.type = OptUpdateType_Adam;
        optUpdate.detail = &optUpdateAdam;

        optUpdateAdam.beta1 = fnet->B1;
        optUpdateAdam.beta2 = fnet->B2;
        optUpdateAdam.eps = fnet->eps;
        optUpdateAdam.t = fnet->time_step_in_opt;
    }
    // note here we use mini_batch_size_in_opt
    optUpdate.mini_batch_size = fnet->mini_batch_size_in_opt;
    optUpdate.momentum = fnet->momentum;
    optUpdate.decay = fnet->decay;
    optUpdate.alpha = math21_ml_function_net_opt_get_alpha(fnet);

    int i;
    for (i = 1; i <= fnet->n_node; ++i) {

        mlfunction_node *fnode = fnet->nodes[i - 1];

        math21_time_index_add();

//        if(i==fnet->n_node){
//            printf("ud time end: %d\n", math21_time_index_get());
//        }

        if (fnode->update) {
            fnode->update(fnode, &optUpdate);
        }

        math21_ml_function_debug_node_save_state(fnode);

    }
}

float math21_ml_function_net_train_one_mini_batch_in_function(mlfunction_net *fnet) {
    fnet->n_seen += fnet->mini_batch_size;
    fnet->is_train = 1;
    math21_ml_function_net_forward(fnet);
    math21_ml_function_net_backward(fnet);
    float error = fnet->cost;
    if (fnet->n_seen % fnet->mini_batch_size_in_opt == 0) math21_ml_function_net_opt_update(fnet);
    return error;
}

float math21_ml_function_net_train_single(mlfunction_net *fnet, m21data d) {
    int mini_batch_size = fnet->mini_batch_size;
    math21_tool_assert(d.x.nr % mini_batch_size == 0);
    int n_mini_batch = d.x.nr / mini_batch_size;

    int i;
    float sum = 0;
    float *data_x, *data_y;
#ifdef MATH21_FLAG_USE_CPU
    data_x = fnet->data_x_wrapper;
    data_y = fnet->data_y_wrapper;
#else
    data_x = fnet->data_x_cpu;
    data_y = fnet->data_y_cpu;
#endif
    for (i = 0; i < n_mini_batch; ++i) {
        math21_tool_data_get_next_mini_batch(d, mini_batch_size, i * mini_batch_size, data_x, data_y);
        float err = math21_ml_function_net_train_one_mini_batch_in_function(fnet);
        sum += err;
    }
    return sum / (n_mini_batch * mini_batch_size);
}

// return cpu value
// return value used by rnn etc.
// net cost may not set, use fnet instead if needed.
float *math21_ml_function_net_predict_input(mlfunction_net *fnet, float *input) {
    mlfunction_net fnet_bak = *fnet;

#if defined(MATH21_FLAG_USE_CPU)
    fnet->data_x_wrapper = input;
    fnet->data_y_wrapper = 0;
#else
    fnet->data_x_cpu = input;
    fnet->data_y_cpu = 0;
#endif

    fnet->is_train = 0;
    math21_ml_function_net_forward(fnet);
    float *out;
#if defined(MATH21_FLAG_USE_CPU)
    out = fnet->y_wrapper;
#else
    math21_ml_function_net_pull_output_wrapper(fnet);
    out = fnet->y_cpu;
#endif
    *fnet = fnet_bak;
    return out;
}

void math21_ml_function_net_log_opt_paras(mlfunction_net *fnet) {
    fprintf(stderr, "alpha: %g, momentum: %g, decay: %g\n", fnet->alpha, fnet->momentum, fnet->decay);
}

void math21_ml_function_net_set_mbs(mlfunction_net *fnet, int mbs) {
    fnet->mini_batch_size = mbs;
    int i;
    for (i = 0; i < fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->set_mbs) {
            fnode->set_mbs(fnode, mbs);
        }
    }
}

float *math21_ml_function_net_predict_image(mlfunction_net *fnet, m21image m) {
    m21image image = math21_image_resize_with_padding(m, fnet->data_x_dim[1], fnet->data_x_dim[2]);
    math21_ml_function_net_set_mbs(fnet, 1);
    float *y = math21_ml_function_net_predict_input(fnet, image.data);
    math21_image_destroy_image(&image);
    return y;
}

// no time
void math21_ml_function_net_data_feed(mlfunction_net *fnet, const float *x, const float *y) {
    float *net_data_x;
    float *net_data_y;
#ifdef MATH21_FLAG_USE_CPU
    net_data_x = fnet->data_x_wrapper;
    net_data_y = fnet->data_y_wrapper;
#else
    net_data_x = fnet->data_x_cpu;
    net_data_y = fnet->data_y_cpu;
#endif
    if (x) {
        math21_vector_assign_from_vector_cpu(fnet->data_x_size * fnet->mini_batch_size, x, 1, net_data_x, 1);
    }
    if (y) {
        math21_vector_assign_from_vector_cpu(fnet->data_y_size * fnet->mbs_y, y, 1, net_data_y, 1);
    }
    if (math21_ml_function_net_getlogLevel() > 100) {
        math21_tensor_3d_float_log_cpu("data_x", net_data_x, fnet->n_time_step_in_rnn,
                                       fnet->mini_batch_size / fnet->n_time_step_in_rnn, fnet->data_x_size);
        math21_tensor_2d_float_log_cpu("data_y", net_data_y, fnet->mbs_y, fnet->data_y_size);
        exit(0);
    }
}

NumN netLogLevel = 0;

NumN math21_ml_function_net_getlogLevel() {
    return netLogLevel;
}

void math21_ml_function_net_setlogLevel(NumN logLevel) {
    netLogLevel = logLevel;
}

void math21_ml_function_net_log(const mlfunction_net *fnet) {
    fprintf(stdout, "net architecture\n");
    int i;
    for (i = 1; i <= fnet->n_node; ++i) {
        fprintf(stdout, "%5d ", i);
        mlfunction_node *fnode = fnet->nodes[i - 1];
        if (fnode->log) {
            fnode->log(fnode, "*/summary");
        }
    }
}

void math21_ml_function_net_node_log_data_x(const mlfunction_net *fnet) {
    if (fnet->n_time_step_in_rnn == 1) {
        math21_tensor_2d_float_log_wrapper("data_x", fnet->data_x_wrapper,
                                           fnet->mini_batch_size / fnet->n_time_step_in_rnn, fnet->data_x_size);
    } else {
        math21_tensor_3d_float_log_wrapper("data_x", fnet->data_x_wrapper, fnet->n_time_step_in_rnn,
                                           fnet->mini_batch_size / fnet->n_time_step_in_rnn, fnet->data_x_size);
    }
}

void math21_ml_function_net_node_log_data_y(const mlfunction_net *fnet) {
    math21_tensor_2d_float_log_wrapper("data_y", fnet->data_y_wrapper, fnet->mbs_y, fnet->data_y_size);
}

void math21_ml_function_net_node_log_by_name(const mlfunction_net *fnet, NumN nodeId, const char *varName) {
    m21log("node", nodeId);
    m21log("varName", varName);
    // if graph, the nodeId of data_x may not be zero.
    if (nodeId == 0) {
        if (math21_string_is_equal(varName, "data_x")) {
            math21_ml_function_net_node_log_data_x(fnet);
        } else if (math21_string_is_equal(varName, "data_y")) {
            math21_ml_function_net_node_log_data_y(fnet);
        }
    } else {
        if (nodeId > fnet->n_node) {
            return;
        }
        auto fnode = fnet->nodes[nodeId - 1];
        if (fnode->log) {
            fnode->log(fnode, varName);
        }
    }
}

const void *math21_ml_function_net_node_get_data_x_to_cpu(const mlfunction_net *fnet) {
#if defined(MATH21_FLAG_USE_CPU)
    return fnet->data_x_wrapper;
#else
    return fnet->data_x_cpu;
#endif
}

const void *math21_ml_function_net_node_get_data_y_to_cpu(const mlfunction_net *fnet) {
#if defined(MATH21_FLAG_USE_CPU)
    return fnet->data_y_wrapper;
#else
    return fnet->data_y_cpu;
#endif
}

// todo: abstract these functions
m21rawtensor math21_ml_function_net_getRawTensorToCpu(mlfunction_net *f, const char *varName) {
    auto *f_detail = (mlfunction_net_detail *) f->detail;
//    std::string _varNameNew;
//    if (!math21_ml_function_tool_varName_check(f->name, varName, _varNameNew)) {
//        m21rawtensor rawtensor={0};
//        return rawtensor;
//    }
//    varName = _varNameNew.c_str();
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

const void *math21_ml_function_net_node_get_data_to_cpu(mlfunction_net *fnet, NumN nodeId, const char *varName) {
    m21log("node", nodeId);
    m21log("varName", varName);
    // if graph, the nodeId of data_x may not be zero.
    if (nodeId == 0) {
        if (math21_string_is_equal(varName, "data_x")) {
            return math21_ml_function_net_node_get_data_x_to_cpu(fnet);
        } else if (math21_string_is_equal(varName, "data_y")) {
            return math21_ml_function_net_node_get_data_y_to_cpu(fnet);
        }
    } else {
        if (nodeId <= fnet->n_node) {
            auto fnode = fnet->nodes[nodeId - 1];
            if (fnode->getDataToCpu) {
                return fnode->getDataToCpu(fnode, varName);
            }
        }
    }
    return 0;
}

m21rawtensor math21_ml_function_net_node_get_rawtensor_to_cpu(mlfunction_net *fnet, NumN nodeId, const char *varName) {
    m21log("node", nodeId);
    m21log("varName", varName);
    // if graph, the nodeId of data_x may not be zero.
    if (nodeId == 0) {
        return math21_ml_function_net_getRawTensorToCpu(fnet, varName);
    } else {
        if (nodeId <= fnet->n_node) {
            auto fnode = fnet->nodes[nodeId - 1];
            if (fnode->getDataToCpu) {
                return fnode->getRawTensorToCpu(fnode, varName);
            }
        }
    }
    m21rawtensor rawtensor = {0};
    return rawtensor;
}