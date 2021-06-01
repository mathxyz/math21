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

int math21_vector_argequal_int(int *a, int val, int n);

float math21_vector_norm_2_float(const float *a, int n);

NumN math21_type_get_vector_float_c(const float *v);

NumN math21_type_get_vector_char_c(const char *v);

// text to vector
int *math21_vector_read_from_file_int(const char *filename, size_t *read);

float *math21_vector_read_from_file_float(const char *filename, size_t *read);

NumN math21_rawtensor_size(int *shape);

void math21_rawtensor_shape_set(NumN size, int *shape);

void math21_rawtensor_shape_assign(int *y_dim, const int *x_dim);

NumB math21_vector_is_byte_text(const NumN8 *x, size_t n, NumB isLog);

#ifdef __cplusplus
}
#endif
