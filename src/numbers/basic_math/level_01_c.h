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

int math21_number_min_2_int(int x1, int x2);

int math21_number_min_3_int(int x1, int x2, int x3);

void math21_number_rectangle_resize_just_put_into_box(NumR src_h, NumR src_w,
                                                 NumR box_h, NumR box_w,
                                                 NumR *dst_h, NumR *dst_w);

void math21_number_interval_intersect_to_int(int a, int b, int *c0, int *d0);

NumN math21_number_axis_insert_pos_check(NumN dims, NumZ pos);

NumN math21_number_container_pos_check(NumN size, NumZ pos);

NumN math21_number_container_stride_get_n(NumN n, NumN stride, NumN offset);

NumN math21_number_container_assign_get_n(NumN n,
                                          NumN n_x, NumN stride1_x, NumN offset1_x,
                                          NumN n_y, NumN stride1_y, NumN offset1_y);

#ifdef __cplusplus
}
#endif
