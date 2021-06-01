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

#include "inner_cc.h"
#include "cost.h"

mlfnode_cost_type math21_ml_function_cost_type_get_type(const char *s) {
    if (strcmp(s, "L2") == 0) return mlfnode_cost_type_L2;
    if (strcmp(s, "masked") == 0) return mlfnode_cost_type_masked;
    if (strcmp(s, "L1") == 0) return mlfnode_cost_type_L1;
    if (strcmp(s, "seg") == 0) return mlfnode_cost_type_seg;
    if (strcmp(s, "smooth") == 0) return mlfnode_cost_type_smooth;
    if (strcmp(s, "wgan") == 0) return mlfnode_cost_type_wgan;
    fprintf(stderr, "Couldn't find cost type %s, going with L2\n", s);
    return mlfnode_cost_type_L2;
}

const char *math21_ml_function_cost_type_get_name(mlfnode_cost_type a) {
    switch (a) {
        case mlfnode_cost_type_L2:
            return "L2";
        case mlfnode_cost_type_masked:
            return "masked";
        case mlfnode_cost_type_L1:
            return "L1";
        case mlfnode_cost_type_seg:
            return "seg";
        case mlfnode_cost_type_smooth:
            return "smooth";
        case mlfnode_cost_type_wgan:
            return "wgan";
    }
    return "L2";
}

void math21_ml_function_cost_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options) {
    const char *type_s = math21_function_option_find_str(options, "type", "sse");
    auto type = math21_ml_function_cost_type_get_type(type_s);
    float scale = math21_function_option_find_float_quiet(options, "scale", 1);
    mlfunction_cost *f = math21_ml_function_cost_create(fnode, finput, type, scale);
    f->ratio = math21_function_option_find_float_quiet(options, "ratio", 0);
    f->noobject_scale = math21_function_option_find_float_quiet(options, "noobj", 1);
    f->thresh = math21_function_option_find_float_quiet(options, "thresh", 0);
    f->smooth = math21_function_option_find_float_quiet(options, "smooth", 0);
}

void math21_ml_function_cost_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (mlfunction_cost *) fnode->function;
    math21_ml_function_cost_saveState(f, file);
}

float math21_ml_function_cost_node_getCost(mlfunction_node *fnode) {
    auto *f = (mlfunction_cost *) fnode->function;
    return *f->cost;
}

void math21_ml_function_cost_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (mlfunction_cost *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_cost_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_cost *) fnode->function;
    math21_ml_function_cost_forward(f, net, finput);
}

void math21_ml_function_cost_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_cost *) fnode->function;
    math21_ml_function_cost_backward(f, net, finput);
}

void math21_ml_function_cost_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_cost *) fnode->function;
    math21_ml_function_cost_log(f, varName);
}

const char *math21_ml_function_cost_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_cost *) fnode->function;
    return f->name;
}

void math21_ml_function_cost_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_cost *) fnode->function;
    fnode->mini_batch_size = f->batch;
    math21_rawtensor_shape_assign(fnode->y_dim, f->y_dim);
    fnode->y_size = f->outputs;
    fnode->y = f->output;
    fnode->dy = f->delta;
}

void math21_ml_function_cost_log(const mlfunction_cost *f, const char *varName) {
    fprintf(stdout, "cost %s: (%d, %d, %d, %d) -> (%d, %d, %d, %d)\n",
            math21_ml_function_cost_type_get_name(f->cost_type),
            f->y_dim[1], f->y_dim[2], f->y_dim[0], f->batch, f->y_dim[1], f->y_dim[2], f->y_dim[0], f->batch);
}

mlfunction_cost *math21_ml_function_cost_create(
        mlfunction_node *fnode, const mlfunction_node *finput,
        mlfnode_cost_type cost_type, float scale) {
    auto *f = (mlfunction_cost *) math21_vector_calloc_cpu(1, sizeof(mlfunction_cost));
    int batch = finput->mini_batch_size;
    int inputs = finput->y_size;
    f->batch = batch;
    f->scale = scale;
    f->inputs = inputs;
    f->outputs = inputs;
    math21_rawtensor_shape_assign(f->y_dim, finput->y_dim);
    f->cost_type = cost_type;
    f->cost = math21_vector_create_with_default_value_cpu(1, 0);

#ifndef MATH21_FLAG_USE_CPU
    f->tmp_cpu = math21_vector_create_with_default_value_cpu(inputs * batch, 0);
#endif
    f->output = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
    f->name = math21_string_create_from_string("cost");
    if (fnode) {
        fnode->type = mlfnode_type_cost;
        fnode->function = f;
        fnode->saveState = math21_ml_function_cost_node_saveState;
        fnode->getCost = math21_ml_function_cost_node_getCost;
        fnode->set_mbs = math21_ml_function_cost_node_set_mbs;
        fnode->forward = math21_ml_function_cost_node_forward;
        fnode->backward = math21_ml_function_cost_node_backward;
        fnode->log = math21_ml_function_cost_node_log;
        fnode->getName = math21_ml_function_cost_node_getName;
        math21_ml_function_cost_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_cost_resize(mlfunction_node *fnode, mlfunction_cost *l, int w, int h) {

    l->y_dim[1] = h;
    l->y_dim[2] = w;
    int inputs = math21_rawtensor_size(l->y_dim);
    l->inputs = inputs;
    l->outputs = inputs;

#ifndef MATH21_FLAG_USE_CPU
    l->tmp_cpu = math21_vector_resize_with_default_value_cpu(l->tmp_cpu, l->outputs * l->batch, 0);
#endif
    l->output = math21_vector_resize_with_default_value_wrapper(l->output, l->outputs * l->batch, 0);
    l->delta = math21_vector_resize_with_default_value_wrapper(l->delta, l->outputs * l->batch, 0);

    if (fnode) {
        math21_ml_function_cost_node_reset(fnode);
    }
}

void math21_ml_function_cost_forward(mlfunction_cost *f, mlfunction_net *net, mlfunction_node *finput) {
    if (math21_vector_isEmpty_wrapper(net->data_y_wrapper)) return;
    if (f->smooth) {
        math21_vector_kx_wrapper(f->batch * f->inputs, (1 - f->smooth), net->data_y_wrapper, 1);
        math21_vector_k_add_x_wrapper(f->batch * f->inputs, f->smooth * 1. / f->inputs, net->data_y_wrapper, 1);
    }

    if (f->cost_type == mlfnode_cost_type_smooth) {
        math21_vector_loss_smooth_l1_wrapper(f->batch * f->inputs, finput->y, net->data_y_wrapper, f->delta,
                                             f->output);
    } else if (f->cost_type == mlfnode_cost_type_L1) {
        math21_vector_loss_l1_wrapper(f->batch * f->inputs, finput->y, net->data_y_wrapper, f->delta, f->output);
    } else if (f->cost_type == mlfnode_cost_type_wgan) {
        math21_tool_assert(0);
    } else {
        math21_vector_loss_l2_wrapper(f->batch * f->inputs, finput->y, net->data_y_wrapper, f->delta, f->output);
    }

    if (f->cost_type == mlfnode_cost_type_seg && f->noobject_scale != 1) {
        math21_vector_kx_by_mask_wrapper(f->batch * f->inputs, f->noobject_scale, f->delta, net->data_y_wrapper, 0);
        math21_vector_kx_by_mask_wrapper(f->batch * f->inputs, f->noobject_scale, f->output, net->data_y_wrapper, 0);
    }
    if (f->cost_type == mlfnode_cost_type_masked) {
        math21_vector_assign_by_mask_wrapper(f->batch * f->inputs, finput->dy, MATH21_MASK_NUM, net->data_y_wrapper, 0);
    }

    if (f->ratio) {
        math21_tool_assert(0);
    }

    if (f->thresh) {
        math21_vector_zero_by_thresh_wrapper(f->batch * f->inputs, f->delta, 1, f->thresh * 1. / f->inputs);
    }

    float *loss;
#if defined(MATH21_FLAG_USE_CPU)
    loss = f->output;
#else
    loss = f->tmp_cpu;
    math21_vector_pull_wrapper(f->output, f->tmp_cpu, f->batch * f->inputs);
#endif
    f->cost[0] = math21_vector_sum_cpu(loss, f->batch * f->inputs);
}


void math21_ml_function_cost_backward(mlfunction_cost *f, mlfunction_net *net, mlfunction_node *finput) {
    math21_vector_kx_add_y_wrapper(f->batch * f->inputs, f->scale, f->delta, 1, finput->dy, 1);
}

void math21_ml_function_cost_saveState(const mlfunction_cost *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
    math21_vector_serialize_c_cpu(file, f->cost, 1);
}
