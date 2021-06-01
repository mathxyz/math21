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

m21array2d math21_array2d_create(NumN nr, NumN nc, NumN type);

m21array2d math21_array2d_concat_vertically(m21array2d m1, m21array2d m2);

void math21_array2d_free(m21array2d m);

m21data math21_tool_data_concat_2(m21data d1, m21data d2);

m21data math21_tool_data_concat_n(m21data *d, int n);

void math21_tool_data_free(m21data d);

void math21_tool_data_get_next_mini_batch(m21data d, int n, int offset, float *x, float *y);

m21data math21_tool_data_get_ith_part(m21data d, int i, int n);

#ifdef __cplusplus
}
#endif
