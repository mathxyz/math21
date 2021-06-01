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

#include "res.h"

void math21_ml_function_res_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                  const mlfunction_node *finput, m21list *options) {
    char *l = math21_function_option_find(options, "from");
    int index = atoi(l);
    int index_fnode = fnode->id - 1;
    if (index < 0) index = index_fnode + index;

    mlfunction_node *from = fnet->nodes[index];

    mlfunction_res *f = math21_ml_function_res_create(fnode, finput->mini_batch_size, index, finput->y_dim[2],
                                                      finput->y_dim[1], finput->y_dim[0], from->y_dim[2],
                                                      from->y_dim[1],
                                                      from->y_dim[0]);

    f->k1 = math21_function_option_find_float_quiet(options, "beta", 1);
    f->k2 = math21_function_option_find_float_quiet(options, "alpha", 1);
    const char *activation_s = math21_function_option_find_str(options, "activation", "linear");
    f->activation = math21_function_activation_get_type(activation_s);
}

void math21_ml_function_res_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_res *f = (mlfunction_res *) fnode->function;
    math21_ml_function_res_saveState(f, file);
}

void math21_ml_function_res_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_res *f = (mlfunction_res *) fnode->function;
    f->batch = mini_batch_size;
}

void math21_ml_function_res_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_res *) fnode->function;
    math21_ml_function_res_forward(f, net, finput);
}

void math21_ml_function_res_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_res *f = (mlfunction_res *) fnode->function;
    math21_ml_function_res_backward(f, net, finput);
}

void math21_ml_function_res_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_res *) fnode->function;
    math21_ml_function_res_log(f, varName);
}

const char *math21_ml_function_res_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_res *) fnode->function;
    return f->name;
}

void math21_ml_function_res_node_reset(mlfunction_node *fnode) {
    mlfunction_res *f = (mlfunction_res *) fnode->function;
    fnode->mini_batch_size = f->batch;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->y_size = fnode->y_dim[0] * fnode->y_dim[1] * fnode->y_dim[2];
    fnode->y = f->output;
    fnode->dy = f->delta;
}

void math21_ml_function_res_log(const mlfunction_res *f, const char *varName) {
    fprintf(stdout, "%s: (%d, %d, %d, %d) -> (%d, %d, %d, %d), index = %d\n",
            f->name,
            f->h, f->w, f->c, f->batch,
            f->out_h, f->out_w, f->out_c, f->batch,
            f->index
    );
}

// shortcut
mlfunction_res *
math21_ml_function_res_create(mlfunction_node *fnode, int batch, int index, int w, int h, int c, int w2, int h2,
                              int c2) {
    mlfunction_res *l = (mlfunction_res *) math21_vector_calloc_cpu(1, sizeof(mlfunction_res));
    l->batch = batch;
    l->w = w2;
    l->h = h2;
    l->c = c2;
    l->out_w = w;
    l->out_h = h;
    l->out_c = c;
    l->outputs = w * h * c;
    l->inputs = l->outputs;
    l->index = index;
    l->output = math21_vector_create_with_default_value_wrapper(l->outputs * batch, 0);
    l->delta = math21_vector_create_with_default_value_wrapper(l->outputs * batch, 0);
    l->name = math21_string_create_from_string("res");
    if (fnode) {
        fnode->type = mlfnode_type_res;
        fnode->function = l;
        fnode->saveState = math21_ml_function_res_node_saveState;
        fnode->set_mbs = math21_ml_function_res_node_set_mbs;
        fnode->forward = math21_ml_function_res_node_forward;
        fnode->backward = math21_ml_function_res_node_backward;
        fnode->log = math21_ml_function_res_node_log;
        fnode->getName = math21_ml_function_res_node_getName;
        math21_ml_function_res_node_reset(fnode);
    }
    return l;
}

void math21_ml_function_res_resize(mlfunction_node *fnode, mlfunction_res *l, int w, int h) {
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w * h * l->out_c;
    l->inputs = l->outputs;
    l->output = math21_vector_resize_with_default_value_wrapper(l->output, l->outputs * l->batch, 0);
    l->delta = math21_vector_resize_with_default_value_wrapper(l->delta, l->outputs * l->batch, 0);
    if (fnode) {
        math21_ml_function_res_node_reset(fnode);
    }
}

// Z = h(Y), Y = k1*X1 + k2*X2
void math21_ml_function_res_forward(mlfunction_res *l, mlfunction_net *net, mlfunction_node *finput) {
    if (net->is_train) {
        math21_vector_set_wrapper(l->batch * l->outputs, 0, l->delta, 1);
    }
    // Y <- X2
    math21_vector_assign_from_vector_wrapper(l->outputs * l->batch, finput->y, 1, l->output, 1);

    // Y <- k1*X1 + k2*Y
    math21_vector_feature2d_add_2_wrapper(l->batch,
                                          l->k1, net->nodes[l->index]->y, l->c, l->h, l->w,
                                          l->k2, l->output, l->out_c, l->out_h, l->out_w);

    // Z = h(Y)
    math21_function_activation_vector_wrapper(l->output, l->outputs * l->batch, l->activation);
}

void math21_ml_function_res_backward(mlfunction_res *l, mlfunction_net *net, mlfunction_node *finput) {
    // dL/dY = dL/dZ *.ele h.d(Y)
    math21_function_activation_gradient_vector_wrapper(l->output, l->outputs * l->batch, l->activation, l->delta);

    math21_tool_assert_to_do_remove(1);
    // dL/dX2 += k2*dL/dY
    if (1) {
        math21_vector_kx_add_y_wrapper(l->outputs * l->batch, l->k2, l->delta, 1, finput->dy, 1);
    }

    // by ye
    // dL/dX2 += k2*dL/dY
    // dL/dX2 += dL/dY
    if (0) {
        math21_vector_kx_add_y_wrapper(l->outputs * l->batch, 1, l->delta, 1, finput->dy, 1);
        // dL/dX2 += (k2 -1)*dL/dY
        math21_vector_feature2d_add_3_wrapper(
                l->batch,
                0, net->nodes[l->index]->y, l->c, l->h, l->w,
                (l->k2 - 1), l->delta, l->out_c, l->out_h, l->out_w,
                1, finput->dy, l->out_c, l->out_h, l->out_w
        );
    }
    // dL/dX1 += k1*dL/dY
    math21_vector_feature2d_add_2_wrapper(l->batch,
                                          l->k1, l->delta, l->out_c, l->out_h, l->out_w,
                                          1, net->nodes[l->index]->dy, l->c, l->h, l->w);
}

void math21_ml_function_res_saveState(const mlfunction_res *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->batch * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->batch * f->outputs);
}