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

typedef struct {
    int n; // number of classes
    char **names; // name of each class
} mllabels;

// (x, y) is box center
struct mlbox {
    float x, y, w, h; // here x = ic, y = ir, todo: use ir, ic
};

typedef struct mlbox mlbox;

struct m21LabelBox{
    NumN id;
    float x, y, w, h;
};

typedef struct m21LabelBox m21LabelBox;

struct m21rectangle {
    int top;
    int bottom;
    int left;
    int right;
};

typedef struct m21rectangle m21rectangle;

struct mldbox {
    float dx, dy, dw, dh;
};

typedef struct mldbox mldbox;

struct mldetection {
    mlbox bbox;
    int classes;
    float *prob;
    float *mask; // will remove
    float objectness;
    int sort_class;
};

typedef struct mldetection mldetection;

struct mlobject {
    int id;
    mlbox bbox;
    float prob;
};

typedef struct mlobject mlobject;

void math21_ml_box_detection_to_results(NumN ndetections, const mldetection *detections,
                                        NumN nresults, mlobject *results);

void math21_ml_box_log_object(mlobject *results, int n);

void math21_ml_box_do_nms_obj(mldetection *detections, int ndetections, float thresh);

void math21_ml_box_do_nms_sort(mldetection *dets, int total, int classes, float thresh);

float math21_ml_box_rmse(mlbox a, mlbox b);

mlbox math21_ml_box_vector_to_box(float *f, int stride);

float math21_ml_box_iou(mlbox a, mlbox b);

// todo: set random color
void math21_ml_box_draw_detections(m21image image, mldetection *detections, int ndetections, float thresh);

#ifdef __cplusplus
}
#endif
