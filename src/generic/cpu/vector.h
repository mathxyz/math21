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

NumB math21_generic_vector_is_equal_cpu(NumN n, const void *x, const void *y,
                                        NumR epsilon, NumN logLevel, NumN type);

NumR math21_generic_vector_distance_cpu(NumN n, const void *x, const void *y, NumR norm, NumN type);

NumR math21_generic_vector_max_cpu(NumN n, const void *x, NumN type);

NumR math21_generic_vector_min_cpu(NumN n, const void *x, NumN type);

void math21_generic_tensor_reverse_axis_3_in_d3_cpu(void *x, NumN d1, NumN d2, NumN d3, NumN type);

#ifdef __cplusplus
}
#endif
