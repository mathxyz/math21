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

void math21_time_index_add();

int math21_time_index_get();

int math21_time_index_get_start_time();

int math21_time_index_get_end_time();

int math21_time_index_get_debug_time();

NumB math21_time_is_debug();

void math21_time_set_debug(NumN debugLevel);

void math21_error(const char *s);

void math21_file_error(const char *s);

void math21_file_warn(const char *s);

time_t math21_c_tim(time_t *timer);

void math21_c_seed_random_generator(unsigned int seed);

void math21_c_seed_random_generator_by_current_time();

unsigned int math21_c_seed_get();

NumB math21_global_tensor_is_log_all_elements();

void math21_global_tensor_enable_log_all_elements();

NumB math21_global_tensor_is_log_matlab_style();

void math21_global_tensor_enable_log_matlab_style();

NumB math21_global_tensor_is_log_no_last_new_line();

void math21_global_tensor_enable_log_no_last_new_line();

NumN math21_global_ad_debug_var_id();

void math21_global_set_ad_debug_var_id(NumN id);

NumB math21_global_ad_log_data();

NumB math21_global_is_debug();

NumB math21_global_ad_is_check_nan();

void math21_global_ad_enable_check_nan();

#ifdef __cplusplus
}
#endif
