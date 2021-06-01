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

#include "node.h"

mlfunction_node *math21_ml_function_node_create() {
    auto *node = (mlfunction_node *)math21_vector_calloc_cpu(1, sizeof(mlfunction_node));
    node->type = mlfnode_type_none;
    return node;
}

// user should set node to 0 after the call.
void math21_ml_function_node_destroy(mlfunction_node *node) {
    if (!node) {
        return;
    }
    math21_vector_free_cpu(node);
}

void math21_ml_function_node_log(const mlfunction_node *f, const char *name) {
    printf("\n");
    if (name) {
        printf("name: %s\n", name);
    }
    printf("id: %d\n", f->id);
    printf("mini_batch_size: %d\n", f->mini_batch_size);
    printf("x_size: %d\n", f->x_size);
    printf("y_size: %d\n", f->y_size);
    int i;
    printf("x_dim: ");
    for (i = 0; i < 3; ++i) {
        printf("%d, ", f->x_dim[i]);
    }
    printf("\n");
    printf("y_dim: ");
    for (i = 0; i < 3; ++i) {
        printf("%d, ", f->y_dim[i]);
    }
    printf("\n");
}

mlfnode_type math21_ml_function_node_type_string_to_type(const char *type) {
    if (strcmp(type, "[conn]") == 0
        || strcmp(type, "[connected]") == 0)
        return mlfnode_type_fully_connected;
    if (strcmp(type, "[local]") == 0) return mlfnode_type_locally_connected;
    if (strcmp(type, "[conv]") == 0
        || strcmp(type, "[convolutional]") == 0)
        return mlfnode_type_conv;
    if (strcmp(type, "[deconv]") == 0
        || strcmp(type, "[deconvolutional]") == 0)
        return mlfnode_type_deconv;
    if (strcmp(type, "[batchnorm]") == 0) return mlfnode_type_batchnorm;
    if (strcmp(type, "[avg]") == 0
        || strcmp(type, "[avgpool]") == 0)
        return mlfnode_type_average_pooling;
    if (strcmp(type, "[max]") == 0
        || strcmp(type, "[maxpool]") == 0)
        return mlfnode_type_max_pooling;
    if (strcmp(type, "[soft]") == 0
        || strcmp(type, "[softmax]") == 0)
        return mlfnode_type_softmax;
    if (strcmp(type, "[cluster_pool]") == 0)return mlfnode_type_cluster_pooling;
    if (strcmp(type, "[upsample]") == 0) return mlfnode_type_sample;
    if (strcmp(type, "[shortcut]") == 0) return mlfnode_type_res;
    if (strcmp(type, "[route]") == 0) return mlfnode_type_route;
    if (strcmp(type, "[yolo]") == 0) return mlfnode_type_yolo;
    if (strcmp(type, "[cost]") == 0) return mlfnode_type_cost;
    if (strcmp(type, "[lstm]") == 0) return mlfnode_type_lstm;
    if (strcmp(type, "[rnn]") == 0) return mlfnode_type_rnn;
    if (strcmp(type, "[gru]") == 0) return mlfnode_type_gru;
    if (strcmp(type, "[net]") == 0
        || strcmp(type, "[network]") == 0)
        return mlfnode_type_net;
    if (strcmp(type, "[dropout]") == 0) return mlfnode_type_dropout;
    return mlfnode_type_none;
}
