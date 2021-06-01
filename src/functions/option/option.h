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

typedef struct {
    char *type;
    m21list *options;
} m21section;

typedef struct {
    char *key;
    char *val;
    int used; // can use multiple times.
} m21kvp;

void math21_function_option_insert(m21list *l, char *key, char *val);

int math21_function_option_read(char *s, m21list *options);

m21list *math21_function_option_cfg_read(const char *filename);

void math21_function_option_free_section(m21section *s);

char *math21_function_option_find(m21list *l, const char *key);

const char *math21_function_option_find_str(m21list *l, const char *key, const char *def);

const char *math21_function_option_find_str_quiet(m21list *l, const char *key, const char *def);

int math21_function_option_find_int(m21list *l, const char *key, int def);

int math21_function_option_find_int_quiet(m21list *l, const char *key, int def);

NumN math21_function_option_find_NumN_quiet(m21list *l, const char *key, NumN def);

float math21_function_option_find_float_quiet(m21list *l, const char *key, float def);

float math21_function_option_find_float(m21list *l, const char *key, float def);

void math21_function_option_log_unused(m21list *l);

#ifdef __cplusplus
}
#endif
