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

NumB math21_ml_is_function_paras_file(const char *filename);

void math21_ml_function_net_parse_options(mlfunction_net *fnet, m21list *options);

void math21_ml_function_net_save_function_paras_upto(mlfunction_net *fnet, const char *filename, int index_node_cutoff);

void math21_ml_function_net_save_function_paras(mlfunction_net *fnet, const char *filename);

void math21_ml_function_net_load_function_paras_from_config_upto(mlfunction_net *fnet, const char *filename,
                                                                 int index_node_start,
                                                                 int index_node_cutoff);

void math21_ml_function_net_load_function_paras_from_config(mlfunction_net *fnet, const char *filename);

mlfunction_net *math21_ml_function_net_load_function_form_from_config(const char *filename);

mlfunction_net *
math21_ml_function_net_create_from_file(const char *function_form, const char *function_paras, int isClear);

void math21_ml_function_net_reset_rnn_state_when_gpu(mlfunction_net *fnet, int b);

int math21_ml_function_net_resize(mlfunction_net *fnet, int w, int h);

#ifdef __cplusplus
}
#endif
