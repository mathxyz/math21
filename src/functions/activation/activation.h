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

typedef enum{
    MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, MATH21_FUNCTION_ACTIVATION_TYPE_RELU, MATH21_FUNCTION_ACTIVATION_TYPE_RELIE, MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR, MATH21_FUNCTION_ACTIVATION_TYPE_RAMP, MATH21_FUNCTION_ACTIVATION_TYPE_TANH, MATH21_FUNCTION_ACTIVATION_TYPE_PLSE, MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY, MATH21_FUNCTION_ACTIVATION_TYPE_ELU, MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY, MATH21_FUNCTION_ACTIVATION_TYPE_STAIR, MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN, MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN, MATH21_FUNCTION_ACTIVATION_TYPE_SELU
} MATH21_FUNCTION_ACTIVATION_TYPE;

char *math21_function_activation_get_string(MATH21_FUNCTION_ACTIVATION_TYPE a);

MATH21_FUNCTION_ACTIVATION_TYPE math21_function_activation_get_type(const char *s);

void math21_function_activation_vector_wrapper(PointerFloatWrapper x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a);

void math21_function_activation_gradient_vector_wrapper(PointerFloatInputWrapper y, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, PointerFloatWrapper dy);

#ifdef __cplusplus
}
#endif
