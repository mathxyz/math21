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

    // concatenate: xs -> y, split: y -> xs
    // size(d2s) = n_xs
    template<typename T>
    void _math21_vector_concatenate_or_split_axis_2_in_d2_cpu(T **xs, T *y, NumN n_xs,
                                                              NumN d1, const NumN *d2s, NumB isSplit = 0) {
        NumN d2_y = math21_vector_compute_sum_cpu(d2s, n_xs);
        NumN *index_x2y = new NumN[n_xs];
        {
            NumN j;
            index_x2y[0] = 0;
            for (j = 1; j < n_xs; ++j) {
                index_x2y[j] = index_x2y[j - 1] + d2s[j - 1];
            }
        }

        NumN *index_y2x = new NumN[d2_y];
        {
            NumN i, j, k;
            k = 0;
            for (i = 0; i < n_xs; ++i) {
                NumN d2 = d2s[i];
                for (j = 0; j < d2; ++j) {
                    index_y2x[k] = i;
                    ++k;
                }
            }
        }

        size_t size = d1 * d2_y;
        NumN id = 0;
        while (1) {
            if (id >= size) break;
            NumN i1, i2, ix, iy, i2_y;
            iy = id;
            xj_index_1d_to_2d(&i1, &i2_y, iy, d1, d2_y);
            NumN ix_class = index_y2x[i2_y];
            NumN iy_seen = index_x2y[ix_class];
            i2 = i2_y - iy_seen;
            T *x = xs[ix_class];
            NumN d2 = d2s[ix_class];
            xj_index_2d_to_1d(i1, i2, &ix, d1, d2);
            if (!isSplit) {
                y[iy] = x[ix];
            } else {
                x[ix] = y[iy];
            }
            ++id;
        }
        delete[] index_x2y;
        delete[] index_y2x;
    }

    template<typename T>
    void math21_vector_concatenate_axis_2_in_d2_cpu(const T **xs, T *y, NumN n_xs, NumN d1, const NumN *d2s) {
        _math21_vector_concatenate_or_split_axis_2_in_d2_cpu((T **) xs, y, n_xs, d1, d2s, 0);
    }

    template<typename T>
    void math21_vector_split_axis_2_in_d2_cpu(T **xs, const T *y, NumN n_xs, NumN d1, const NumN *d2s) {
        _math21_vector_concatenate_or_split_axis_2_in_d2_cpu(xs, (T *) y, n_xs, d1, d2s, 1);
    }

    template<typename T>
    void math21_vector_linear_cpu(NumR k1, const T *x, NumR k2, const T *y, T *z, NumN n) {
        NumN size = n;
        NumN id = 0;
        while (1) {
            if (id >= size) return;
            z[id] = k1 * x[id] + k2 * y[id];
            ++id;
        }
    }
}