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

#include "inner_cc.h"
#include "level_01.h"
#include "level_01_c.h"

#if !defined(MATH21_FLAG_USE_CUDA)

#include "level_01"

#endif

namespace math21 {
    void math21_number_get_from_and_to(NumN n0, NumZ &from, NumZ &to) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        }
        if (to == 0) {
            to = n;
        } else if (to < 0) {
            to = n + to + 1;
        }
    }

    // 1 <= from <= to <= n
    NumB math21_number_check_from_and_to(NumN n, NumZ from, NumZ to) {
        if (from < 1 || to > n || from > to) {
            return 0;
        } else {
            return 1;
        }
    }

    void math21_number_get_from_only_with_check(NumN n0, NumZ &from) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        } else {
            MATH21_ASSERT(from <= n)
        }
    }

    void math21_number_get_from_and_num_with_check(NumN n0, NumZ &from, NumN &num) {
        NumZ n = n0;
        if (from == 0) {
            from = 1;
        } else if (from < 0) {
            from = n + from + 1;
        } else {
            MATH21_ASSERT(from <= n)
        }
        if (num == 0) {
            num = (NumN) n + 1 - from;
        } else {
            MATH21_ASSERT(from + num <= n + 1)
        }
    }

}

using namespace math21;

int math21_number_min_2_int(int x1, int x2) {
    return xjmin(x1, x2);
}

int math21_number_min_3_int(int x1, int x2, int x3) {
    return xjmin(x1, x2, x3);
}

// keep aspect ratio
void math21_number_rectangle_resize_just_put_into_box(NumR src_nr, NumR src_nc,
                                                 NumR box_nr, NumR box_nc,
                                                 NumR *dst_nr, NumR *dst_nc) {
    math21_tool_assert(dst_nr && dst_nc);
    NumR ratio = xjmin(box_nc / src_nc, box_nr / src_nr);
    *dst_nr = src_nr * ratio;
    *dst_nc = src_nc * ratio;
}

// [c, d] <- [a, b] intersect [c, d]
template<typename T>
void _math21_number_interval_intersect_to(T a, T b, T *c0, T *d0) {
    int c = *c0;
    int d = *d0;
    if (c < a) c = a;
    if (c > b) c = b;
    if (d < a) d = a;
    if (d > b) d = b;
    *c0 = c;
    *d0 = d;
}

// [c, d] <- [a, b] intersect [c, d]
void math21_number_interval_intersect_to_int(int a, int b, int *c0, int *d0) {
    _math21_number_interval_intersect_to(a, b, c0, d0);
}

NumN math21_number_axis_insert_pos_check(NumN dims, NumZ pos) {
    MATH21_ASSERT(xjIsIn(xjabs(pos), 1, dims + 1),
                  "abs|pos| not in [1, n+1], n = " << dims << ", pos = " << pos);
    NumN i;
    if (pos < 0) {
        i = dims + 2 + pos;
    } else {
        i = pos;
    }
    return i;
}

NumN math21_number_container_pos_check(NumN size, NumZ pos) {
    MATH21_ASSERT(xjIsIn(xjabs(pos), 1, size),
                  "|pos| not in [1, n], n = " << size << ", pos = " << pos);
    NumN i;
    if (pos < 0) {
        i = size + 1 + pos;
    } else {
        i = (NumN) pos;
    }
    return i;
}

NumN math21_number_container_stride_get_n(NumN n, NumN stride, NumN offset) {
    return (NumN) xjceil((n - offset) / (NumR) stride);
}

NumN math21_number_container_assign_get_n(NumN n,
                                          NumN n_x, NumN stride1_x, NumN offset1_x,
                                          NumN n_y, NumN stride1_y, NumN offset1_y) {
    NumN n_max_x;
    n_max_x = math21_number_container_stride_get_n(n_x, stride1_x, offset1_x);
    NumN n_max_y;
    n_max_y = math21_number_container_stride_get_n(n_y, stride1_y, offset1_y);
    NumN n_max = xjmin(n_max_x, n_max_y);
    if (n == 0)n = n_max;
    return xjmin(n_max, n);
}