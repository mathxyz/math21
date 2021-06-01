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

namespace math21 {
    // todo: remove
    // swap axes 2 and 4 in dim5 tensor
    // (d1, d2, d3, d4, d5) -> (d1, d4, d3, d2, d5)
    template<typename T>
    void math21_vector_swap_axes_24_in_d5_cpu_z(const T *x, T *y, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        size_t size = d1 * d2 * d3 * d4 * d5;
        NumN id = 0;
        while (1) {
            if (id >= size) return;
            NumN i1, i2, i3, i4, i5, ix, iy, d2_y, d4_y, i2_y, i4_y;
            iy = id;
            d4_y = d2;
            d2_y = d4;
            i5 = iy % d5;
            iy = iy / d5;
            i4_y = iy % d4_y;
            iy = iy / d4_y;
            i3 = iy % d3;
            iy = iy / d3;
            i2_y = iy % d2_y;
            i1 = iy / d2_y;

            i2 = i4_y;
            i4 = i2_y;
            ix = i1 * d2 * d3 * d4 * d5 + i2 * d3 * d4 * d5 + i3 * d4 * d5 + i4 * d5 + i5;
            iy = id;
            y[iy] = x[ix];
            ++id;
        }
    }

// todo: deprecate math21_vector_sum(const float ...), use math21_vector_compute_sum_cpu instead
    template<typename T>
    T math21_vector_compute_sum_cpu(const T *v, int n) {
        int i;
        T sum = 0;
        for (i = 0; i < n; ++i) sum += v[i];
        return sum;
    }

    template<typename T>
    T math21_vector_compute_mul_cpu(const T *v, int n) {
        int i;
        T sum = 1;
        for (i = 0; i < n; ++i) sum *= v[i];
        return sum;
    }

    template<typename T>
    T math21_operator_container_compute_mul_brackets(const T *v, int n) {
        int i;
        T sum = 1;
        for (i = 1; i <= n; ++i) sum *= v[i];
        return sum;
    }

    template<typename T>
    void math21_vector_repeat_axis_2_in_d3_cpu(const T *x, T *y, NumN d1, NumN d2, NumN d3, NumN repeat) {
        size_t size = d1 * d2 * repeat * d3;
        NumN id = 0;
        while (1) {
            if (id >= size) return;
            NumN i1, i2, i3, ix, iy, d2_y, i2_y;
            iy = id;
            d2_y = repeat * d2;
            xj_index_1d_to_3d(&i1, &i2_y, &i3, iy, d1, d2_y, d3);
            i2 = i2_y / repeat;
            xj_index_3d_to_1d(i1, i2, i3, &ix, d1, d2, d3);
            y[iy] = x[ix];
            ++id;
        }
    }

    // size(repeats) = d2
    template<typename T>
    void math21_vector_repeats_axis_2_in_d3_cpu(const T *x, T *y, NumN d1, NumN d2, NumN d3, const NumN *repeats) {
        NumN d2_y = math21_vector_compute_sum_cpu(repeats, d2);
        NumN *index_y2x = new NumN[d2_y];
        {
            NumN i, j, k;
            k = 0;
            for (i = 0; i < d2; ++i) {
                NumN repeat = repeats[i];
                for (j = 0; j < repeat; ++j) {
                    index_y2x[k] = i;
                    ++k;
                }
            }
        }

        size_t size = d1 * d2_y * d3;
        NumN id = 0;
        while (1) {
            if (id >= size) break;
            NumN i1, i2, i3, ix, iy, i2_y;
            iy = id;
            xj_index_1d_to_3d(&i1, &i2_y, &i3, iy, d1, d2_y, d3);
            i2 = index_y2x[i2_y];
            xj_index_3d_to_1d(i1, i2, i3, &ix, d1, d2, d3);
            y[iy] = x[ix];
            ++id;
        }
        delete[] index_y2x;
    }

    template<typename T>
    void math21_vector_sum_undo_repeat_axis_2_in_d3_cpu(T *x, const T *y, NumN d1, NumN d2, NumN d3, NumN repeat,
                                                        NumB clear = 1) {
        size_t size = d1 * d2 * d3;
        NumN id = 0;
        while (1) {
            if (id >= size) return;
            NumN i1, i2, i3, ix, iy, d2_y, i2_y;
            ix = id;
            xj_index_1d_to_3d(&i1, &i2, &i3, ix, d1, d2, d3);
            d2_y = repeat * d2;
            if (clear) {
                x[ix] = 0;
            }
            NumN j;
            i2_y = i2 * repeat;
            for (j = 0; j < repeat; ++j) {
                xj_index_3d_to_1d(i1, i2_y, i3, &iy, d1, d2_y, d3);
                x[ix] += y[iy];
                ++i2_y;
            }
            ++id;
        }
    }

// size(repeats) = d2
    template<typename T>
    void
    math21_vector_sum_undo_repeats_axis_2_in_d3_cpu(T *x, const T *y, NumN d1, NumN d2, NumN d3, const NumN *repeats,
                                                    NumB clear = 1) {
        NumN d2_y = math21_vector_compute_sum_cpu(repeats, d2);
//        NumN index_x2y[d2];
        NumN *index_x2y = new NumN[d2];
        {
            NumN j;
            index_x2y[0] = 0;
            for (j = 1; j < d2; ++j) {
                index_x2y[j] = index_x2y[j - 1] + repeats[j - 1];
            }
        }

        size_t size = d1 * d2 * d3;
        NumN id = 0;
        while (1) {
            if (id >= size) break;
            NumN i1, i2, i3, ix, iy, i2_y;
            ix = id;
            xj_index_1d_to_3d(&i1, &i2, &i3, ix, d1, d2, d3);
            if (clear) {
                x[ix] = 0;
            }
            NumN j;
            i2_y = index_x2y[i2];
            for (j = 0; j < repeats[i2]; ++j) {
                xj_index_3d_to_1d(i1, i2_y, i3, &iy, d1, d2_y, d3);
                x[ix] += y[iy];
                ++i2_y;
            }
            ++id;
        }
        delete[] index_x2y;
    }

}