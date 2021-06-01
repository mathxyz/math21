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

#include "sample.h"

void math21_ml_function_sample_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                     const mlfunction_node *finput, m21list *options) {

    int stride = math21_function_option_find_int(options, "stride", 2);
    mlfunction_sample *f = math21_ml_function_sample_create(fnode, finput->mini_batch_size, finput->y_dim[2],
                                                            finput->y_dim[1], finput->y_dim[0], stride);
    f->scale = math21_function_option_find_float_quiet(options, "scale", 1);
}

void math21_ml_function_sample_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_sample *f = (mlfunction_sample *) fnode->function;
    math21_ml_function_sample_saveState(f, file);
}

void math21_ml_function_sample_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_sample *f = (mlfunction_sample *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_sample_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_sample *) fnode->function;
    math21_ml_function_sample_forward(f, finput, net->is_train);
}

void math21_ml_function_sample_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_sample *f = (mlfunction_sample *) fnode->function;
    math21_ml_function_sample_backward(f, finput);
}

void math21_ml_function_sample_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_sample *) fnode->function;
    math21_ml_function_sample_log(f, varName);
}

const char *math21_ml_function_sample_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_sample *) fnode->function;
    return f->name;
}

void math21_ml_function_sample_node_reset(mlfunction_node *fnode) {
    mlfunction_sample *f = (mlfunction_sample *) fnode->function;
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

void math21_ml_function_sample_log(const mlfunction_sample *f, const char *varName) {
    fprintf(stdout, "%s                %2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", f->name, f->stride, f->w, f->h, f->c,
            f->out_w, f->out_h, f->out_c);
}

mlfunction_sample *
math21_ml_function_sample_create(mlfunction_node *fnode, int mini_batch_size, int nc, int nr, int nch, int stride) {
    mlfunction_sample *f = (mlfunction_sample *)math21_vector_calloc_cpu(1, sizeof(mlfunction_sample));
    f->batch = mini_batch_size;
    f->w = nc;
    f->h = nr;
    f->c = nch;
    f->out_w = nc * stride;
    f->out_h = nr * stride;
    f->out_c = nch;
    if (stride < 0) {
        stride = -stride;
        f->reverse = 1;
        f->out_w = nc / stride;
        f->out_h = nr / stride;
    }
    f->stride = stride;
    f->outputs = f->out_w * f->out_h * f->out_c;
    f->inputs = f->w * f->h * f->c;
    f->output = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
    if (f->reverse) {
        f->name = math21_string_create_from_string("sumdownsample");
    } else {
        f->name = math21_string_create_from_string("upsample");
    }
    if (fnode) {
        fnode->type = mlfnode_type_sample;
        fnode->function = f;
        fnode->saveState = math21_ml_function_sample_node_saveState;
        fnode->set_mbs = math21_ml_function_sample_node_set_mbs;
        fnode->forward = math21_ml_function_sample_node_forward;
        fnode->backward = math21_ml_function_sample_node_backward;
        fnode->log = math21_ml_function_sample_node_log;
        fnode->getName = math21_ml_function_sample_node_getName;
        math21_ml_function_sample_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_sample_resize(mlfunction_node *fnode, mlfunction_sample *f, int nc, int nr) {
    f->w = nc;
    f->h = nr;
    f->out_w = nc * f->stride;
    f->out_h = nr * f->stride;
    if (f->reverse) {
        f->out_w = nc / f->stride;
        f->out_h = nr / f->stride;
    }
    f->outputs = f->out_w * f->out_h * f->out_c;
    f->inputs = f->h * f->w * f->c;
    f->output = math21_vector_resize_with_default_value_wrapper(f->output, f->outputs * f->batch, 0);
    f->delta = math21_vector_resize_with_default_value_wrapper(f->delta, f->outputs * f->batch, 0);
    if (fnode) {
        math21_ml_function_sample_node_reset(fnode);
    }
}

void math21_ml_function_sample_forward(mlfunction_sample *f, mlfunction_node *finput, NumB is_train) {
    if (is_train) {
        math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
    }
    math21_vector_set_wrapper(f->outputs * f->batch, 0, f->output, 1);
    if (f->reverse) {
        math21_vector_feature2d_sample_wrapper(
                f->batch, f->output, f->c, f->out_h,
                f->out_w, f->stride, 0, f->scale, finput->y);
    } else {
        math21_vector_feature2d_sample_wrapper(f->batch, finput->y, f->c, f->h, f->w, f->stride, 1, f->scale,
                                               f->output);
    }
}

void math21_ml_function_sample_backward(mlfunction_sample *f, mlfunction_node *finput) {
    if (f->reverse) {
        math21_vector_feature2d_sample_wrapper(f->batch, f->delta, f->c, f->out_h, f->out_w, f->stride, 1, f->scale,
                                               finput->dy);
    } else {
        math21_vector_feature2d_sample_wrapper(f->batch, finput->dy, f->c, f->h, f->w, f->stride, 0, f->scale,
                                               f->delta);
    }
}

void math21_ml_function_sample_saveState(const mlfunction_sample *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
}