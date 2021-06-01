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

#include "max_pooling_cpu.h"
#include "max_pooling_cuda.h"
#include "max_pooling_opencl.h"
#include "max_pooling.h"

void math21_ml_function_max_pooling_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                          const mlfunction_node *finput, m21list *options) {
    int stride = math21_function_option_find_int(options, "stride", 1);
    int size = math21_function_option_find_int(options, "size", stride);
    int padding = math21_function_option_find_int_quiet(options, "padding", size - 1);

    int mini_batch_size, nr, nc, nch;
    nch = finput->y_dim[0];
    nr = finput->y_dim[1];
    nc = finput->y_dim[2];
    mini_batch_size = finput->mini_batch_size;

    if (!(nch && nr && nc)) math21_error("Layer before maxpool layer must output image.");

    math21_ml_function_max_pooling_create(fnode, mini_batch_size, nch, nr, nc, size, stride, padding);
}

void math21_ml_function_max_pooling_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_max_pooling *f = (mlfunction_max_pooling *) fnode->function;
    math21_ml_function_max_pooling_saveState(f, file);
}

void math21_ml_function_max_pooling_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_max_pooling *f = (mlfunction_max_pooling *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_max_pooling_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_max_pooling *) fnode->function;
    math21_ml_function_max_pooling_forward(f, finput, net->is_train);
}

void
math21_ml_function_max_pooling_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_max_pooling *f = (mlfunction_max_pooling *) fnode->function;
    math21_ml_function_max_pooling_backward(f, finput);
}

void math21_ml_function_max_pooling_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_max_pooling *) fnode->function;
    math21_ml_function_max_pooling_log(f, varName);
}

const char *math21_ml_function_max_pooling_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_max_pooling *) fnode->function;
    return f->name;
}

void math21_ml_function_max_pooling_node_reset(mlfunction_node *fnode) {
    mlfunction_max_pooling *f = (mlfunction_max_pooling *) fnode->function;
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

void math21_ml_function_max_pooling_log(const mlfunction_max_pooling *f, const char *varName) {
    fprintf(stdout, "%s         %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", f->name, f->size, f->size, f->stride, f->w, f->h,
            f->c, f->out_w, f->out_h, f->out_c);
}

mlfunction_max_pooling *
math21_ml_function_max_pooling_create(mlfunction_node *fnode, int batch, int c, int h, int w, int size, int stride,
                                      int padding) {
    mlfunction_max_pooling *f = (mlfunction_max_pooling *) math21_vector_calloc_cpu(1, sizeof(mlfunction_max_pooling));
    f->batch = batch;
    f->c = c;
    f->h = h;
    f->w = w;
    f->padding = padding;
    f->out_w = (w + padding - size) / stride + 1;
    f->out_h = (h + padding - size) / stride + 1;
    f->out_c = c;
    f->outputs = f->out_h * f->out_w * f->out_c;
    f->inputs = h * w * c;
    f->size = size;
    f->stride = stride;
    f->indexes = math21_vector_create_from_cpuvector_int_wrapper(f->batch * f->outputs, 0, 1);
    f->output = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(f->batch * f->outputs, 0);
    f->name = math21_string_create_from_string("max pooling");
    if (fnode) {
        fnode->type = mlfnode_type_max_pooling;
        fnode->function = f;
        fnode->saveState = math21_ml_function_max_pooling_node_saveState;
        fnode->set_mbs = math21_ml_function_max_pooling_node_set_mbs;
        fnode->forward = math21_ml_function_max_pooling_node_forward;
        fnode->backward = math21_ml_function_max_pooling_node_backward;
        fnode->log = math21_ml_function_max_pooling_node_log;
        fnode->getName = math21_ml_function_max_pooling_node_getName;
        math21_ml_function_max_pooling_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_max_pooling_resize(mlfunction_node *fnode, mlfunction_max_pooling *l, int w, int h) {

    l->h = h;
    l->w = w;
    l->inputs = h * w * l->c;

    l->out_w = (w + l->padding - l->size) / l->stride + 1;
    l->out_h = (h + l->padding - l->size) / l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;

    l->indexes = math21_vector_resize_with_default_value_int_wrapper(l->indexes, l->outputs * l->batch, 0);
    l->output = math21_vector_resize_with_default_value_wrapper(l->output, l->outputs * l->batch, 0);
    l->delta = math21_vector_resize_with_default_value_wrapper(l->delta, l->outputs * l->batch, 0);

    if (fnode) {
        math21_ml_function_max_pooling_node_reset(fnode);
    }
}

void math21_ml_function_max_pooling_forward(mlfunction_max_pooling *f, const mlfunction_node *finput, int is_train) {
    if (is_train) {
        math21_vector_set_wrapper(f->batch * f->outputs, 0, f->delta, 1);
    }
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_max_pooling_forward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_max_pooling_forward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_max_pooling_forward_opencl(f, finput);
#endif
}

void math21_ml_function_max_pooling_backward(mlfunction_max_pooling *f, mlfunction_node *finput) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_max_pooling_backward_cpu(f, finput);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_max_pooling_backward_cuda(f, finput);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_max_pooling_backward_opencl(f, finput);
#endif
}

void math21_ml_function_max_pooling_saveState(const mlfunction_max_pooling *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
//    math21_vector_serialize_int_c_wrapper(file, f->indexes, f->batch * f->outputs);
}
