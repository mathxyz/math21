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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void math21_matrix_transpose(NumR32 *old, int rows, int cols);

void math21_tensor_1d_float_log_cpu(const char *name, const NumR32 *data, NumN d1);

void math21_tensor_2d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2);

void math21_tensor_3d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3);

void math21_tensor_4d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4);

void
math21_tensor_5d_float_log_cpu(const char *name, const NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5);

void math21_tensor_1d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1);

void math21_tensor_2d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2);

void math21_tensor_3d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3);

void
math21_tensor_4d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3, NumN d4);

void
math21_tensor_5d_float_log_wrapper(const char *name, PointerFloatInputWrapper data, NumN d1, NumN d2, NumN d3, NumN d4,
                                   NumN d5);

void math21_rawtensor_log_cpu(const char *name, m21rawtensor rawtensor);

void math21_rawtensor_log_wrapper(const char *name, m21rawtensor rawtensor);

#ifdef __cplusplus
}
#endif
