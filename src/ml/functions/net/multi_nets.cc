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

#include "../files_c.h"
#include "multi_nets.h"

void math21_ml_function_multinet_sync_node_merge_to(mlfunction_node *fnode, mlfunction_node *fbase) {
    if (fnode->type == mlfnode_type_conv) {
        mlfunction_conv *f = (mlfunction_conv *) fnode->function;
        mlfunction_conv *fb = (mlfunction_conv *) fbase->function;
        math21_ml_function_conv_merge_to(f, fb);
    } else if (fnode->type == mlfnode_type_fully_connected) {
        mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
        mlfunction_fully_connected *fb = (mlfunction_fully_connected *) fbase->function;
        math21_ml_function_fully_connected_merge_to(f, fb);
    }
}

void math21_ml_function_multinet_sync_node_scale(mlfunction_node *fnode, float s) {
    if (fnode->type == mlfnode_type_conv) {
        mlfunction_conv *f = (mlfunction_conv *) fnode->function;
        math21_ml_function_conv_scale(f, s);
    } else if (fnode->type == mlfnode_type_fully_connected) {
        mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
        math21_ml_function_fully_connected_scale(f, s);
    }
}

void math21_ml_function_multinet_sync_node_pull(mlfunction_node *fnode) {
    if (fnode->type == mlfnode_type_conv) {
        mlfunction_conv *f = (mlfunction_conv *) fnode->function;
        math21_ml_function_conv_pull_wrapper(f, 0); //Why doesn't rolling_mean be pulled?
    } else if (fnode->type == mlfnode_type_fully_connected) {
        mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
        math21_ml_function_fully_connected_pull_wrapper(f, 0);// Todo: Does scales need to be pulled?
    }
}

void math21_ml_function_multinet_sync_node_distribute(mlfunction_node *fnode, mlfunction_node *fbase) {
    if (fnode->type == mlfnode_type_conv) {
        mlfunction_conv *f = (mlfunction_conv *) fnode->function;
        mlfunction_conv *fb = (mlfunction_conv *) fbase->function;
        math21_ml_function_conv_push_by_wrapper(f, fb, 0); //Why doesn't rolling_mean be pushed?
    } else if (fnode->type == mlfnode_type_fully_connected) {
        mlfunction_fully_connected *f = (mlfunction_fully_connected *) fnode->function;
        mlfunction_fully_connected *fb = (mlfunction_fully_connected *) fbase->function;
        math21_ml_function_fully_connected_push_by_wrapper(f, fb, 0);
    }
}

void math21_ml_function_multinet_sync_ith_node(mlfunction_net **fnets, int n, int j) {
    mlfunction_net *fnet = fnets[0];
    mlfunction_node *fbase = fnet->nodes[j];
    int i;
    for (i = 0; i < n; ++i) {
        math21_gpu_set_device_wrapper(fnets[i]->gpuDevice);
        mlfunction_node *fnode = fnets[i]->nodes[j];
        math21_ml_function_multinet_sync_node_pull(fnode);
        if (i > 0) {
            math21_ml_function_multinet_sync_node_merge_to(fnode, fbase);
        }
    }
    if (n > 1)math21_ml_function_multinet_sync_node_scale(fbase, 1. / n);
    for (i = 0; i < n; ++i) {
        math21_gpu_set_device_wrapper(fnets[i]->gpuDevice);
        mlfunction_node *fnode = fnets[i]->nodes[j];
        math21_ml_function_multinet_sync_node_distribute(fnode, fbase);
    }
}

typedef struct {
    mlfunction_net *fnet;
    m21data d;
    float *err;
} m21_net_train_args;

void *math21_ml_function_net_train_in_thread(void *ptr) {
    m21_net_train_args args = *(m21_net_train_args *) ptr;
    free(ptr);
    math21_gpu_set_device_wrapper(args.fnet->gpuDevice);
    *args.err = math21_ml_function_net_train_single(args.fnet, args.d);
    return 0;
}

void *math21_ml_function_net_get_thread_for_training(mlfunction_net *fnet, m21data d, float *err) {
    m21_net_train_args *ptr = (m21_net_train_args *) math21_vector_calloc_cpu(1, sizeof(m21_net_train_args));
    ptr->fnet = fnet;
    ptr->d = d;
    ptr->err = err;
    void *t = math21_tool_thread_create();
    math21_tool_thread_start(t, 0, math21_ml_function_net_train_in_thread, ptr);
    return t;
}

typedef struct {
    mlfunction_net **fnets;
    int n;
    int j;
} m21_multinet_sync_args;

void *math21_ml_function_multinet_sync_node_in_thread(void *ptr) {
    m21_multinet_sync_args args = *(m21_multinet_sync_args *) ptr;
    free(ptr);
    math21_ml_function_multinet_sync_ith_node(args.fnets, args.n, args.j);
    return 0;
}

void *math21_ml_function_multinet_get_thread_for_sync_node(mlfunction_net **fnets, int n, int j) {
    m21_multinet_sync_args *ptr = (m21_multinet_sync_args *) math21_vector_calloc_cpu(1, sizeof(m21_multinet_sync_args));
    ptr->fnets = fnets;
    ptr->n = n;
    ptr->j = j;
    void *t = math21_tool_thread_create();
    math21_tool_thread_start(t, 0, math21_ml_function_multinet_sync_node_in_thread, ptr);
    return t;
}

void math21_ml_function_multinet_sync(mlfunction_net **fnets, int num_net, int interval) {
    mlfunction_net *fnet = fnets[0];
    void **threads = (void **) math21_vector_calloc_cpu(fnet->n_node, sizeof(void *));

    fnet->n_seen += interval * (num_net - 1) * fnet->mini_batch_size_in_opt;
    int j;
    for (j = 0; j < num_net; ++j) {
        fnets[j]->n_seen = fnet->n_seen;
    }

    for (j = 0; j < fnet->n_node; ++j) {
        threads[j] = math21_ml_function_multinet_get_thread_for_sync_node(fnets, num_net, j);
    }
    for (j = 0; j < fnet->n_node; ++j) {
        math21_tool_thread_join_and_destroy(threads[j], 0);
    }
    free(threads);
}

float math21_ml_function_multinets_train(mlfunction_net **fnets, int n_net, m21data d, int interval) {
    math21_tool_assert(fnets[0]->mini_batch_size_in_opt * n_net == d.x.nr);
    void **threads = (void **) math21_vector_calloc_cpu(n_net, sizeof(void *));
    float *errors = (float *) math21_vector_calloc_cpu(n_net, sizeof(float));

    float sum = 0;
    int i;
    for (i = 0; i < n_net; ++i) {
        m21data p = math21_tool_data_get_ith_part(d, i, n_net);
        threads[i] = math21_ml_function_net_get_thread_for_training(fnets[i], p, errors + i);
    }
    for (i = 0; i < n_net; ++i) {
        math21_tool_thread_join_and_destroy(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    if (math21_ml_function_net_get_update_count(fnets[0]) % interval == 0) {
        if (n_net > 1)printf("Syncing... ");
        fflush(stdout);
        math21_ml_function_multinet_sync(fnets, n_net, interval);
        if (n_net > 1)printf("Done!\n");
    }
    free(threads);
    free(errors);
    return sum / (n_net);
}
