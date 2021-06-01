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
#include "route.h"

// route specified convolutional layers
void math21_ml_function_route_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                    const mlfunction_node *finput, m21list *options) {
    char *l = math21_function_option_find(options, "layers");
    int len = strlen(l);
    if (!l) math21_error("Route Layer must specify input layers");
    int num_layer = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++num_layer;
    }

    int *input_layers = (int *) math21_vector_calloc_cpu(num_layer, sizeof(int));
    for (i = 0; i < num_layer; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        int index_fnode = fnode->id - 1;
        if (index < 0) index = index_fnode + index;
        input_layers[i] = index;
    }
    math21_ml_function_route_create(fnode, fnet, finput->mini_batch_size, num_layer, input_layers);
}

void math21_ml_function_route_node_saveState(const mlfunction_node *fnode, FILE *file) {
    mlfunction_route *f = (mlfunction_route *) fnode->function;
    math21_ml_function_route_saveState(f, file);
}

void math21_ml_function_route_node_set_mbs(mlfunction_node *fnode, int mini_batch_size) {
    fnode->mini_batch_size = mini_batch_size;
    mlfunction_route *f = (mlfunction_route *) fnode->function;
    f->mini_batch_size = mini_batch_size;
}

void math21_ml_function_route_node_forward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    auto *f = (mlfunction_route *) fnode->function;
    math21_ml_function_route_forward(f, net);
}

void math21_ml_function_route_node_backward(mlfunction_node *fnode, mlfunction_net *net, mlfunction_node *finput) {
    mlfunction_route *f = (mlfunction_route *) fnode->function;
    math21_ml_function_route_backward(f, net);
}

void math21_ml_function_route_node_log(const mlfunction_node *fnode, const char *varName) {
    auto *f = (const mlfunction_route *) fnode->function;
    math21_ml_function_route_log(f, varName);
}

const char *math21_ml_function_route_node_getName(const mlfunction_node *fnode) {
    auto *f = (const mlfunction_route *) fnode->function;
    return f->name;
}

void math21_ml_function_route_node_reset(mlfunction_node *fnode) {
    mlfunction_route *f = (mlfunction_route *) fnode->function;
    fnode->mini_batch_size = f->mini_batch_size;
    fnode->y_dim[0] = f->out_c;
    fnode->y_dim[1] = f->out_h;
    fnode->y_dim[2] = f->out_w;
    fnode->x_size = f->inputs;
    fnode->y_size = f->outputs;
    fnode->y = f->output;
    fnode->dy = f->delta;
}

void math21_ml_function_route_destroy(mlfunction_route *f) {
    math21_vector_free_cpu(f->input_layers);
    math21_vector_free_cpu(f->input_sizes);
    math21_vector_free_wrapper(f->output);
    math21_vector_free_wrapper(f->delta);
    math21_vector_free_cpu(f);
}

void math21_ml_function_route_log(const mlfunction_route *f, const char *varName) {
    fprintf(stdout, "%s  ", f->name);
    fprintf(stdout, " %d", f->input_layers[0]);
    int i;
    for (i = 1; i < f->num_layer; ++i) {
        fprintf(stdout, " %d", f->input_layers[i]);
    }
    fprintf(stdout, "\n");
}

// todo: free input_layers
// y has shape (nch, nr, nc)
mlfunction_route *
math21_ml_function_route_create(mlfunction_node *fnode, const mlfunction_net *net, int mini_batch_size, int num_layer,
                                int *input_layers) {
    mlfunction_route *f = (mlfunction_route *) math21_vector_calloc_cpu(1, sizeof(mlfunction_route));
    f->mini_batch_size = mini_batch_size;
    f->num_layer = num_layer;
    f->input_layers = input_layers;
    f->input_sizes = (int *) math21_vector_calloc_cpu(num_layer, sizeof(int));

    int i;
    const mlfunction_node *first = net->nodes[f->input_layers[0]];
//    math21_ml_function_node_log(first, "first");
    f->out_c = first->y_dim[0];
    f->out_h = first->y_dim[1];
    f->out_w = first->y_dim[2];
    f->outputs = first->y_size;
    f->input_sizes[0] = first->y_size;
    for (i = 1; i < f->num_layer; ++i) {
        int index = f->input_layers[i];
        const mlfunction_node *next = net->nodes[index];
//        math21_ml_function_node_log(next, "next");
        f->outputs += next->y_size;
        f->input_sizes[i] = next->y_size;
        if (next->y_dim[1] == first->y_dim[1] && next->y_dim[2] == first->y_dim[2]) {
            f->out_c += next->y_dim[0];
        } else {
            printf("%d %d, %d %d\n", next->y_dim[1], next->y_dim[2], first->y_dim[1], first->y_dim[2]);
            math21_error("route fail"); // ye add
            f->out_h = f->out_w = f->out_c = 0;
        }
    }
    f->inputs = f->outputs;
    f->output = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);
    f->delta = math21_vector_create_with_default_value_wrapper(mini_batch_size * f->outputs, 0);

    f->name = math21_string_create_from_string("route");
    if (fnode) {
        fnode->type = mlfnode_type_route;
        fnode->function = f;
        fnode->saveState = math21_ml_function_route_node_saveState;
        fnode->set_mbs = math21_ml_function_route_node_set_mbs;
        fnode->forward = math21_ml_function_route_node_forward;
        fnode->backward = math21_ml_function_route_node_backward;
        fnode->log = math21_ml_function_route_node_log;
        fnode->getName = math21_ml_function_route_node_getName;
        math21_ml_function_route_node_reset(fnode);
    }
    return f;
}

void math21_ml_function_route_resize(mlfunction_node *fnode, mlfunction_route *f, const mlfunction_net *net) {
    int i;
    const mlfunction_node *first = net->nodes[f->input_layers[0]];
    f->out_c = first->y_dim[0];
    f->out_h = first->y_dim[1];
    f->out_w = first->y_dim[2];
    f->outputs = first->y_size;
    f->input_sizes[0] = first->y_size;
    for (i = 1; i < f->num_layer; ++i) {
        int index = f->input_layers[i];
        const mlfunction_node *next = net->nodes[index];
        f->outputs += next->y_size;
        f->input_sizes[i] = next->y_size;
        if (next->y_dim[1] == first->y_dim[1] && next->y_dim[2] == first->y_dim[2]) {
            f->out_c += next->y_dim[0];
        } else {
            printf("%d %d, %d %d\n", next->y_dim[1], next->y_dim[2], first->y_dim[1], first->y_dim[2]);
            math21_error("route fail"); // ye add
            f->out_h = f->out_w = f->out_c = 0;
        }
    }
    f->inputs = f->outputs;

    f->output = math21_vector_resize_with_default_value_wrapper(f->output, f->outputs * f->mini_batch_size, 0);
    f->delta = math21_vector_resize_with_default_value_wrapper(f->delta, f->outputs * f->mini_batch_size, 0);
    if (fnode) {
        math21_ml_function_route_node_reset(fnode);
    }
}

// concatenate convolution features
// Y = (X1, Xn), Y = (y1, yb), Xi = (xi1, xib), yb = (x1b, xnb)
void math21_ml_function_route_forward(mlfunction_route *f, mlfunction_net *net) {
    if (net->is_train) {
        math21_vector_set_wrapper(f->mini_batch_size * f->outputs, 0, f->delta, 1);
    }
    int ilayer, imb;
    int offset = 0;
    for (ilayer = 0; ilayer < f->num_layer; ++ilayer) {
        int index = f->input_layers[ilayer];
        auto input = net->nodes[index]->y;
        int input_size = f->input_sizes[ilayer];
        // Y <- Xi
        for (imb = 0; imb < f->mini_batch_size; ++imb) {
            // yb <- xib
            // x and y have same stride because of same shape order.
            math21_vector_assign_from_vector_wrapper(input_size, input + imb * input_size, 1,
                                                     f->output + offset + imb * f->outputs, 1);
        }
        offset += input_size;
    }
}

// (dX1, dXn) += dY, dY = (dy1, dyb), dXi = (dxi1, dxib), (dx1b, dxnb) += dyb
void math21_ml_function_route_backward(mlfunction_route *f, mlfunction_net *net) {
    int ilayer, imb;
    int offset = 0;
    for (ilayer = 0; ilayer < f->num_layer; ++ilayer) {
        int index = f->input_layers[ilayer];
        PointerFloatWrapper delta = net->nodes[index]->dy;
        int input_size = f->input_sizes[ilayer];
        for (imb = 0; imb < f->mini_batch_size; ++imb) {
            math21_vector_kx_add_y_wrapper(input_size, 1, f->delta + offset + imb * f->outputs, 1,
                                           delta + imb * input_size, 1);
        }
        offset += input_size;
    }
}

void math21_ml_function_route_saveState(const mlfunction_route *f, FILE *file) {
    math21_vector_serialize_c_wrapper(file, f->output, f->mini_batch_size * f->outputs);
    math21_vector_serialize_c_wrapper(file, f->delta, f->mini_batch_size * f->outputs);
}