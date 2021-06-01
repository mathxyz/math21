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

void math21_ml_function_softmax_tree_wrapper(PointerFloatWrapper input, int in_class_size, int mini_batch_size, int stride, float temp, PointerFloatWrapper output, m21tree hier);

void math21_ml_function_softmax_wrapper(PointerFloatWrapper input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride, float temp, PointerFloatWrapper output);

void math21_ml_function_softmax_x_ent_wrapper(int n, PointerFloatWrapper pred, PointerFloatWrapper truth, PointerFloatWrapper delta, PointerFloatWrapper error);

#ifdef __cplusplus
}
#endif
