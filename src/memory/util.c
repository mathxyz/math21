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

#include "util_c.h"

void *math21_vector_memset_cpu(void *s, int c, size_t n) {
    void *p_data = memset(s, c, n);
    if (!p_data) {
        math21_error("memset error!");
    }
    return p_data;
}

void *math21_vector_memcpy_cpu(void *dst, const void *src, size_t n) {
    void *p_data = memcpy(dst, src, n);
    if (!p_data) {
        math21_error("memcpy error!");
    }
    return p_data;
}

void *math21_vector_malloc_cpu(size_t size) {
    void *p_data = malloc(size);
    if (!p_data) {
        printf(SIZE_T_SPECIFIER "\n", size);
        math21_error("malloc error!");
    }
    return p_data;
}

// better to use math21_vector_create_with_default_value_cpu.
void *math21_vector_calloc_cpu(size_t nmemb, size_t size) {
    void *p_data = calloc(nmemb, size);
    if (!p_data) {
        printf(SIZE_T_SPECIFIER "\n", size);
        math21_error("calloc error!");
    }
    return p_data;
}

void *math21_vector_realloc_cpu(void *ptr, size_t size) {
    void *p_data = realloc(ptr, size);
    if (!p_data) {
        printf(SIZE_T_SPECIFIER "\n", size);
        math21_error("realloc error!");
    }
    return p_data;
}

void math21_memory_free_cpu(void *ptr) {
    free(ptr);
}
