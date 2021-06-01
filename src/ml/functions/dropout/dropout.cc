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
#include "dropout_cpu.h"
#include "dropout_cuda.h"
#include "dropout_opencl.h"
#include "dropout.h"

void math21_ml_function_dropout_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                      mlfunction_node *finput, m21list *options) {

    float probability = math21_function_option_find_float(options, "probability", .5);
    const char *name = math21_function_option_find_str_quiet(options, "name", 0);
    int n_time_step = math21_function_option_find_int_quiet(options, "n_time_step", 1);
    math21_ml_function_dropout_create(fnode, finput, probability, n_time_step, name);
}

void math21_ml_function_dropout_node_saveState(const mlfunction_node *fnode, FILE *file) {
    auto *f = (mlfunction_dropout *) fnode->function;
    math21_ml_function_dropout_saveState(f, file);
}

void math21_ml_function_dropout_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    auto *f = (mlfunction_dropout *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_dropout_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_dropout *) fnode->function;
    math21_ml_function_dropout_forward(f, finput, net->is_train);
}

void math21_ml_function_dropout_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_dropout *) fnode->function;
    math21_ml_function_dropout_backward(f, finput);
}

void math21_ml_function_dropout_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_dropout *) fnode->function;
    math21_ml_function_dropout_log(f, varName);
}

const char *math21_ml_function_dropout_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_dropout *) fnode->function;
    return f->name;
}

void math21_ml_function_dropout_node_reset(mlfunction_node *fnode) {
    auto *f = (mlfunction_dropout *) fnode->function;
    fnode->mini_batch_size = f->total_mbs;
    math21_rawtensor_shape_assign(fnode->y_dim, f->y_dim);
    fnode->y_size = f->outputs;
    fnode->y = f->y;
    fnode->dy = f->dy;
}

void math21_ml_function_dropout_log(const mlfunction_dropout *f, const char *varName) {
    if (f->n_time_step > 1) {
        fprintf(stdout, "%s with time: (%d, %d, %d, %d, %d) -> (%d, %d, %d, %d, %d), rate = %.2f\n",
                f->name,
                f->y_dim[1], f->y_dim[2], f->y_dim[0], f->n_time_step, f->batch,
                f->y_dim[1], f->y_dim[2], f->y_dim[0], f->n_time_step, f->batch,
                f->rate);
    } else {
        fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), rate = %.2f\n",
                f->name,
                f->y_dim[1], f->y_dim[2], f->y_dim[0], f->total_mbs,
                f->y_dim[1], f->y_dim[2], f->y_dim[0], f->total_mbs, f->rate);
    }
}

// finput can have empty vector, but must have its shape.
mlfunction_dropout *
math21_ml_function_dropout_create(mlfunction_node *fnode, mlfunction_node *finput, float rate, int n_time_step, const char *name) {

    int inputs = finput->y_size;

    auto *f = (mlfunction_dropout *) math21_vector_calloc_cpu(1, sizeof(mlfunction_dropout));
    f->rate = rate;
    f->inputs = inputs;
    f->outputs = inputs;
    f->total_mbs = finput->mini_batch_size;
    f->scale = 1. / (1. - rate);

    math21_rawtensor_shape_assign(f->y_dim, finput->y_dim);
    if(math21_rawtensor_size(f->y_dim)==0){
        math21_rawtensor_shape_set(f->outputs, f->y_dim);
    }

    math21_tool_assert(n_time_step > 0);
    f->n_time_step = n_time_step;
    f->i_time_step = 0;
    math21_tool_assert(f->total_mbs % n_time_step == 0);
    f->batch = f->total_mbs / n_time_step;

    f->rand = math21_vector_create_with_default_value_wrapper(f->batch * f->inputs, 0);
    f->y = math21_vector_create_with_default_value_wrapper(f->total_mbs * f->outputs, 0);
    f->dy = math21_vector_create_with_default_value_wrapper(f->total_mbs * f->outputs, 0);
    if (!name) {
        name = "dropout";
    }
    f->name = math21_string_create_from_string(name);
    if (fnode) {
        fnode->type = mlfnode_type_dropout;
        fnode->function = f;
        fnode->saveState = math21_ml_function_dropout_node_saveState;
        fnode->set_mbs = math21_ml_function_dropout_node_set_mbs;
        fnode->forward = math21_ml_function_dropout_node_forward;
        fnode->backward = math21_ml_function_dropout_node_backward;
        fnode->log = math21_ml_function_dropout_node_log;
        fnode->getName = math21_ml_function_dropout_node_getName;
        math21_ml_function_dropout_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_dropout_forward(mlfunction_dropout *f, mlfunction_node *finput,
                                        int is_train) {
    if (is_train) {
        if (f->i_time_step == 0) {
            math21_vector_set_wrapper(f->total_mbs * f->outputs, 0, f->dy, 1);
        }
    }
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_dropout_forward_cpu(f, finput, is_train);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_dropout_forward_cuda(f, finput, is_train);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_dropout_forward_opencl(f, finput, is_train);
#endif
}

void math21_ml_function_dropout_backward(mlfunction_dropout *f, mlfunction_node *finput) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_dropout_backward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_dropout_backward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_dropout_backward_opencl(f, finput);
#endif
}

void math21_ml_function_dropout_saveState(const mlfunction_dropout *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->y, f->total_mbs * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->dy, f->total_mbs * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->rand, f->total_mbs * f->outputs);
}

void math21_ml_function_dropout_increase_by_time(mlfunction_dropout *f, int time_steps) {
    f->i_time_step += time_steps;
    int num = time_steps * f->batch * f->outputs;
    f->y += num;
    f->dy += num;
}

void math21_ml_function_dropout_reset(mlfunction_dropout *f) {
    int num = f->i_time_step * f->batch * f->outputs;
    f->y -= num;
    f->dy -= num;
    f->i_time_step = 0;
}
