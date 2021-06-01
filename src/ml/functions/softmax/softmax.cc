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

#include "softmax_wrapper.h"
#include "softmax.h"

void math21_ml_function_softmax_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      const mlfunction_node *finput, m21list *options) {
    int groups = math21_function_option_find_int_quiet(options, "groups", 1);
    mlfunction_softmax *f = math21_ml_function_softmax_create(fnode, finput, groups);

    f->temperature = math21_function_option_find_float_quiet(options, "temperature", 1);
    const char *tree_file = math21_function_option_find_str(options, "tree", 0);
    if (tree_file) f->softmax_tree = math21_data_struncture_tree_read(tree_file);
    f->spatial = math21_function_option_find_int_quiet(options, "spatial", 0);
    f->noloss = math21_function_option_find_int_quiet(options, "noloss", 0);
}

void math21_ml_function_softmax_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
    math21_ml_function_softmax_saveState(f, file);
}

float math21_ml_function_softmax_node_getCost(mlfunction_node *fnode) {
    mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
    return *f->cost;
}

void math21_ml_function_softmax_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_softmax_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_softmax *) fnode->function;
    math21_ml_function_softmax_forward(f, net, finput);
}

void math21_ml_function_softmax_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
    math21_ml_function_softmax_backward(f, net, finput);
}

void math21_ml_function_softmax_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_softmax *) fnode->function;
    math21_ml_function_softmax_log(f, varName);
}

const char *math21_ml_function_softmax_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_softmax *) fnode->function;
    return f->name;
}

void math21_ml_function_softmax_node_reset(mlfunction_node *fnode) {
    mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->y_dim[0] = f->c;
    fnode->y_dim[1] = f->h;
    fnode->y_dim[2] = f->w;
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

void math21_ml_function_softmax_log(const mlfunction_softmax *f, const char *varName) {
    fprintf(stdout, "softmax: (%d, %d, %d, %d) -> (%d, %d, %d, %d)\n",
            f->h, f->w, f->c, f->batch, f->h, f->w, f->c, f->batch);
}

mlfunction_softmax *
math21_ml_function_softmax_create(mlfunction_node *fnode, const mlfunction_node *finput, int groups) {
    mlfunction_softmax *f = (mlfunction_softmax *) math21_vector_calloc_cpu(1, sizeof(mlfunction_softmax));
    int batch = finput->mini_batch_size;
    int inputs = finput->y_size;
    assert(inputs % groups == 0);
    f->c = finput->y_dim[0];
    f->h = finput->y_dim[1];
    f->w = finput->y_dim[2];
    f->batch = batch;
    f->groups = groups;
    f->inputs = inputs;
    f->outputs = inputs;
    f->cost = math21_vector_create_with_default_value_cpu(1, 0);
    f->loss = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
#ifndef MATH21_FLAG_USE_CPU
    f->loss_cpu = math21_vector_create_with_default_value_cpu(inputs * batch, 0);
#endif
    f->output = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(inputs * batch, 0);
    f->name = math21_string_create_from_string("softmax");
    if (fnode) {
        fnode->type = mlfnode_type_softmax;
        fnode->function = f;
        fnode->saveState = math21_ml_function_softmax_node_saveState;
        fnode->getCost = math21_ml_function_softmax_node_getCost;
        fnode->set_mbs = math21_ml_function_softmax_node_set_mbs;
        fnode->forward = math21_ml_function_softmax_node_forward;
        fnode->backward = math21_ml_function_softmax_node_backward;
        fnode->log = math21_ml_function_softmax_node_log;
        fnode->getName = math21_ml_function_softmax_node_getName;
        math21_ml_function_softmax_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_softmax_forward(mlfunction_softmax *f, mlfunction_net *net, mlfunction_node *finput) {
    if (net->is_train) {
        math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
    }
    if (f->softmax_tree) {
        math21_ml_function_softmax_tree_wrapper(finput->y, 1, f->batch, f->inputs, f->temperature, f->output,
                                                *f->softmax_tree);
        /*
        int i;
        int count = 0;
        for (i = 0; i < f->softmax_tree->groups; ++i) {
            int group_size = f->softmax_tree->group_size[i];
            softmax(finput->y + count, group_size, f->batch, f->inputs, 1, 0, 1, f->temperature, f->output + count);
            count += group_size;
        }
        */
    } else {
        if (f->spatial) {
            math21_ml_function_softmax_wrapper(finput->y, f->c, f->batch * f->c, f->inputs / f->c, f->w * f->h, 1,
                                               f->w * f->h, 1, f->output);
        } else {
            math21_ml_function_softmax_wrapper(finput->y, f->inputs / f->groups, f->batch, f->inputs, f->groups,
                                               f->inputs / f->groups, 1,
                                               f->temperature, f->output);
        }
    }
    if (!math21_vector_isEmpty_wrapper(net->data_y_wrapper) && !f->noloss) {
        math21_ml_function_softmax_x_ent_wrapper(f->batch * f->inputs, f->output, net->data_y_wrapper, f->delta,
                                                 f->loss);
        if (f->softmax_tree) {
            math21_vector_assign_by_mask_wrapper(f->batch * f->inputs, f->delta, MATH21_MASK_NUM, net->data_y_wrapper,
                                                 0);
            math21_vector_assign_by_mask_wrapper(f->batch * f->inputs, f->loss, MATH21_MASK_NUM, net->data_y_wrapper,
                                                 0);
        }
        float *loss;
#if defined(MATH21_FLAG_USE_CPU)
        loss = f->loss;
#else
        loss = f->loss_cpu;
        math21_vector_pull_wrapper(f->loss, f->loss_cpu, f->batch * f->inputs);
#endif
        // todo: add sum wrapper, remove loss_cpu. See mlfunction_cost
        f->cost[0] = math21_vector_sum_cpu(loss, f->batch * f->inputs);
    }
}

void math21_ml_function_softmax_backward(mlfunction_softmax *f, mlfunction_net *net, mlfunction_node *finput) {
    math21_vector_kx_add_y_wrapper(f->batch * f->inputs, 1, f->delta, 1, finput->dy, 1);
}

void math21_ml_function_softmax_saveState(const mlfunction_softmax *f, FILE *file) {
//    if (math21_time_index_get() == math21_time_index_get_debug_time()) {
//        math21_tool_assert(0);
//    }
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->loss, f->batch * f->outputs);
    math21_vector_serialize_c_cpu(file, f->cost, 1);
}

void math21_ml_function_softmax_net_set_temperature(mlfunction_net *fnet, float t) {
    int i;
    for (i = 0; i < fnet->n_node; ++i) {
        mlfunction_node *fnode = fnet->nodes[i];
        if (fnode->type == mlfnode_type_softmax) {
            mlfunction_softmax *f = (mlfunction_softmax *) fnode->function;
            f->temperature = t;
        }
    }
}
