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

typedef struct m21node {
    void *val;
    struct m21node *next;
    struct m21node *prev;
} m21node;

typedef struct m21list {
    int size;
    m21node *first;
    m21node *last;
} m21list;

m21list *math21_data_structure_list_create();

void math21_data_structure_list_insert(m21list *l, void *val);

void math21_data_structure_list_node_to_last_free(m21node *n);

void math21_data_structure_list_free(m21list *l);

void **math21_data_structure_list_to_array(m21list *l);

void math21_data_structure_free_pp(char **ptrs, int n);

#ifdef __cplusplus
}
#endif
