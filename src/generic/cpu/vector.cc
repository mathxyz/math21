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

#include "template_vector.h"
#include "vector.h"

using namespace math21;

NumB math21_generic_vector_is_equal_cpu(NumN n, const void *x, const void *y,
                                        NumR epsilon, NumN logLevel, NumN type) {
    if (type == m21_type_NumN8) {
        return math21_template_vector_is_equal_cpu(n, (const NumN8 *) x, (const NumN8 *) y, epsilon, logLevel);
    } else if (type == m21_type_NumR) {
        return math21_template_vector_is_equal_cpu(n, (const NumR *) x, (const NumR *) y, epsilon, logLevel);
    } else if (type == m21_type_NumR32) {
        return math21_template_vector_is_equal_cpu(n, (const NumR32 *) x, (const NumR32 *) y, epsilon, logLevel);
    } else if (type == m21_type_NumR64) {
        return math21_template_vector_is_equal_cpu(n, (const NumR64 *) x, (const NumR64 *) y, epsilon, logLevel);
    } else {
        math21_tool_assert(0);
        return 0;
    }
}

NumR math21_generic_vector_distance_cpu(NumN n, const void *x, const void *y, NumR norm, NumN type) {
    if (type == m21_type_NumR) {
        return math21_template_vector_distance(n, (const NumR *) x, (const NumR *) y, norm);
    } else if (type == m21_type_NumR32) {
        return math21_template_vector_distance(n, (const NumR32 *) x, (const NumR32 *) y, norm);
    } else if (type == m21_type_NumR64) {
        return math21_template_vector_distance(n, (const NumR64 *) x, (const NumR64 *) y, norm);
    } else {
        math21_tool_assert(0);
        return 0;
    }
}

NumR math21_generic_vector_max_cpu(NumN n, const void *x, NumN type) {
    NumN i;
    if (type == m21_type_NumR) {
        return math21_template_vector_max(n, (const NumR *) x, i);
    } else if (type == m21_type_NumR32) {
        return math21_template_vector_max(n, (const NumR32 *) x, i);
    } else if (type == m21_type_NumR64) {
        return math21_template_vector_max(n, (const NumR64 *) x, i);
    } else {
        math21_tool_assert(0);
        return 0;
    }
}

NumR math21_generic_vector_min_cpu(NumN n, const void *x, NumN type) {
    NumN i;
    if (type == m21_type_NumR) {
        return math21_template_vector_min(n, (const NumR *) x, i);
    } else if (type == m21_type_NumR32) {
        return math21_template_vector_min(n, (const NumR32 *) x, i);
    } else if (type == m21_type_NumR64) {
        return math21_template_vector_min(n, (const NumR64 *) x, i);
    } else {
        math21_tool_assert(0);
        return 0;
    }
}

void math21_generic_tensor_reverse_axis_3_in_d3_cpu(void *x, NumN d1, NumN d2, NumN d3, NumN type) {
    if (type == m21_type_NumR) {
        math21_template_tensor_reverse_axis_3_in_d3_cpu((NumR *) x, d1, d2, d3);
    } else if (type == m21_type_NumR32) {
        math21_template_tensor_reverse_axis_3_in_d3_cpu((NumR32 *) x, d1, d2, d3);
    } else if (type == m21_type_NumR64) {
        math21_template_tensor_reverse_axis_3_in_d3_cpu((NumR64 *) x, d1, d2, d3);
    } else {
        math21_tool_assert(0);
    }
}