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

#include "vector_cpu.h"
#include "vector.h"
#include "inner_cc.h"

using namespace math21;

int math21_vector_argequal_int(int *a, int val, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (a[i] == val) return i;
    }
    return -1;
}

float math21_vector_norm_2_float(const float *a, int n) {
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

NumN math21_type_get_vector_float_c(const float *v) {
    return m21_type_vector_float_c;
}

NumN math21_type_get_vector_char_c(const char *v) {
    return m21_type_vector_char_c;
}

int *math21_vector_read_from_file_int(const char *filename, size_t *read) {
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    auto *d = static_cast<int *>(math21_vector_calloc_cpu(size, sizeof(int)));
    int n, one;
    one = fscanf(fp, "%d", &n);
    while (one == 1) {
        ++count;
        if (count > size) {
            size = size * 2;
            d = static_cast<int *>(math21_vector_realloc_cpu(d, size * sizeof(int)));
        }
        d[count - 1] = n;
        one = fscanf(fp, "%d", &n);
    }
    fclose(fp);
    d = static_cast<int *>(math21_vector_realloc_cpu(d, count * sizeof(int)));
    *read = count;
    return d;
}

float *math21_vector_read_from_file_float(const char *filename, size_t *read) {
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    auto *d = static_cast<float *>(math21_vector_calloc_cpu(size, sizeof(float)));
    int one;
    float n;
    one = fscanf(fp, "%f", &n);
    while (one == 1) {
        ++count;
        if (count > size) {
            size = size * 2;
            d = static_cast<float *>(math21_vector_realloc_cpu(d, size * sizeof(float)));
        }
        d[count - 1] = n;
        one = fscanf(fp, "%f", &n);
    }
    fclose(fp);
    d = static_cast<float *>(math21_vector_realloc_cpu(d, count * sizeof(float)));
    *read = count;
    return d;
}

NumN math21_rawtensor_size(int *shape) {
    NumN dims = MATH21_DIMS_RAW_TENSOR;
    NumN i;
    NumN size = 1;
    for (i = 0; i < dims; ++i) size *= shape[i];
    return size;
}

void math21_rawtensor_shape_set(NumN size, int *shape) {
    math21_tool_assert(shape);
    NumN dims = MATH21_DIMS_RAW_TENSOR;
    NumN i;
    for (i = 0; i < dims; ++i) {
        shape[i] = 1;
    }
    shape[0] = size;
}

void math21_rawtensor_shape_assign(int *y_dim, const int *x_dim) {
    math21_vector_assign_from_vector_int_cpu(MATH21_DIMS_RAW_TENSOR, x_dim, 1, y_dim, 1);
}

NumB math21_vector_is_byte_text(const NumN8 *x, size_t n, NumB isLog) {
    auto pch = static_cast<const NumN8 *>(memchr(x, 0, n));
    if (pch != NULL) {
        if (isLog) {
            printf("bad byte %d at %d\n", *pch, (NumN)(pch - x + 1));
        }
        return 0;
    }
    return 1;
}
