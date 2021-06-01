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
#include "../box/files_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int math21_ml_function_net_get_detections_num(mlfunction_net *fnet, float thresh);

mldetection *math21_ml_function_net_boxes_create(mlfunction_net *fnet , float thresh, int *ndetections);

void math21_ml_function_net_boxes_destroy(mldetection *dets, int ndetections);

void math21_ml_function_net_boxes_set(mlfunction_net *fnet, int nr, int nc, float thresh, int relative,
                                      mldetection *dets);

mldetection *math21_ml_function_net_boxes_get(
        mlfunction_net *fnet, int nr, int nc, float thresh, int relative, int *ndetections);

#ifdef __cplusplus
}
#endif
