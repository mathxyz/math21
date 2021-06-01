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

#pragma once

#include "inner.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlfunction_yolo mlfunction_yolo;
struct mlfunction_yolo {
    const char *name;
    int is_this_type;
    int batch; // mini_batch_size
    int h; // nr_grids
    int w; // nc_grids
    int c; // num_box*(num_class + 4 + 1);
    int out_h; // nr_grids
    int out_w; // nc_grids
    int out_c; // num_box*(num_class + 4 + 1);
    int n; // num_box
    int total; // total_prior
    int classes; // num_class
    float *cost; // ||dL/dY||^2
    int *mask; // prior_mask with size num_box
    float *biases; // num_box used only, not total_prior
    int outputs; // l.out_h*l.out_w*l.out_c;
    int inputs; // l.out_h*l.out_w*l.out_c;
    int truths;
    float *output; // Y
    PointerFloatGpu output_gpu; // used when gpu and cpu
    float *delta; // dL/dY
    PointerFloatGpu delta_gpu; // used when gpu and cpu

    int max_boxes;
    float ignore_thresh;
    float truth_thresh;
    int onlyforward; // we compute some derivative if onlyforward = 0.
    int *map;

    int net_train;
    float *net_truth; // cpu pointer
    int net_h;
    int net_w;
    int net_index; // net layer index

    float jitter;
    int random;
};

void math21_ml_function_yolo_parse(mlfunction_node *fnode, const mlfunction_net *fnet,
                                   const mlfunction_node *finput, m21list *options);

int *math21_ml_function_yolo_read_map(const char *filename);

void math21_ml_function_yolo_log(const mlfunction_yolo *f, const char *varName);

mlfunction_yolo *
math21_ml_function_yolo_create(mlfunction_node *fnode, int mini_batch_size, int nc_grids, int nr_grids, int num_box,
                               int total_prior,
                               int *prior_mask, int num_class, int max_boxes);

void math21_ml_function_yolo_resize(mlfunction_node *fnode, mlfunction_yolo *l, int w, int h);

void math21_ml_function_yolo_forward(mlfunction_yolo *l, mlfunction_node *finput);

void math21_ml_function_yolo_backward(mlfunction_yolo *l, mlfunction_node *finput);

int math21_ml_function_yolo_num_detections(mlfunction_yolo *l, float thresh);

int math21_ml_function_yolo_get_detections(mlfunction_yolo *l, int w, int h, int netw, int neth, float thresh,
                                           int relative, mldetection *dets);

#ifndef MATH21_FLAG_USE_CPU

void math21_ml_function_yolo_forward_gpu(mlfunction_yolo *l, mlfunction_node *finput);

void math21_ml_function_yolo_backward_gpu(mlfunction_yolo *l, mlfunction_node *finput);

#endif

void math21_ml_function_yolo_saveState(const mlfunction_yolo *f, FILE *file);

#ifdef __cplusplus
}
#endif
