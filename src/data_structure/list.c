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

#include "../matrix/files_c.h"
#include "list_c.h"

m21list *math21_data_structure_list_create() {
    m21list *l = math21_vector_malloc_cpu(sizeof(m21list));
    l->size = 0;
    l->first = 0;
    l->last = 0;
    return l;
}

void math21_data_structure_list_insert(m21list *l, void *val) {
    m21node *node_new = math21_vector_malloc_cpu(sizeof(m21node));
    node_new->val = val;
    node_new->next = 0;

    if (!l->last) {
        l->first = node_new;
        node_new->prev = 0;
    } else {
        l->last->next = node_new;
        node_new->prev = l->last;
    }
    l->last = node_new;
    ++l->size;
}

void math21_data_structure_list_node_to_last_free(m21node *n) {
    m21node *next;
    while (n) {
        next = n->next;
        math21_vector_free_cpu(n);
        n = next;
    }
}

void math21_data_structure_list_free(m21list *l) {
    math21_data_structure_list_node_to_last_free(l->first);
    math21_vector_free_cpu(l);
}

void **math21_data_structure_list_to_array(m21list *l) {
    void **a = math21_vector_calloc_cpu(l->size, sizeof(void *));
    int count = 0;
    m21node *n = l->first;
    while (n) {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

void math21_data_structure_free_pp(char **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}
