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

NumN math21_type_num_to_ten(NumN type);

m21point math21_tensor_1d_create(NumN type, NumN size, void *data, NumB isDataShared);

m21point math21_tensor_nd_create(NumN type, m21point pd, void *data, NumB isDataShared);

// use this, not math21_point_destroy to speed up.
void math21_tensor_destroy(m21point point);

m21point math21_tensor_copy_shape(m21point point);

NumN math21_tensor_size(m21point point);

void *math21_tensor_data(m21point point);

NumB math21_point_is_empty(m21point point);

m21point math21_point_share_assign(m21point point);

// just assign, so can use '=' instead.
m21point math21_point_assign(m21point point);

m21point math21_point_init(m21point point);

m21point math21_point_destroy(m21point point);

void math21_point_log(m21point point);

NumB math21_point_save(m21point point, const char *path);

m21point math21_point_load(const char *path);

NumB math21_point_isEqual(m21point x, m21point y, NumR epsilon);

m21point math21_point_create_by_type(NumN type);

m21point math21_point_create_ad_point_const(m21point tenPoint);

m21point math21_point_create_ad_point_input(m21point tenPoint);

void math21_point_ad_point_set_value(m21point adPoint, m21point tenPoint);

m21point math21_point_ad_point_get_value(m21point adPoint);

void math21_destroy();

#ifdef __cplusplus
}
#endif
