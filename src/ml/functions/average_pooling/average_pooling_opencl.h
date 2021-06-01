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

#include "average_pooling.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_OPENCL

void math21_ml_function_average_pooling_forward_opencl(mlfunction_average_pooling *f, const mlfunction_node*finput);

void math21_ml_function_average_pooling_backward_opencl(mlfunction_average_pooling *f, mlfunction_node *finput);

#endif

#ifdef __cplusplus
}
#endif
