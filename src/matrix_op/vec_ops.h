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

    struct less_than_id_cluster {
        const VecN &v;

        less_than_id_cluster(const VecN &v) : v(v) {

        }

        bool operator()(int i1, int i2) {
            return (v(i1 + 1) < v(i2 + 1));
        }
    };


    template<typename Compare>
    std::vector<int> &math21_operator_sort_indexes(NumN size, std::vector<int> &idx, Compare comp) {
        if (idx.size() != size) {
            math21_tool_std_vector_resize(idx, size);
        }
        for (unsigned int i = 0; i < size; i++) {
            idx[i] = i;
        }

        sort(idx.begin(), idx.end(), comp);
        return idx;
    }


    template<typename T1, typename T2>
    void math21_operator_vec_2_vec(const std::vector<T1> &A, Tensor<T2> &B) {
        if (B.isSameSize(A.size()) == 0) {
            B.setSize(A.size());
        }
        for (NumN i = 1; i <= B.size(); ++i) {
            B(i) = (T2) A[i - 1];
        }
    }

    template<typename T>
    void math21_operator_array_2_vec(const Array<T> &x, Tensor<T> &y) {
        NumN n = x.size();
        MATH21_ASSERT(n == y.size());
        for (NumN i = 1; i <= n; i++) {
            y(i) = x(i);
        }
    }

    template<typename T>
    void math21_operator_vec_2_array(const Tensor<T> &x, Array<T> &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
        for (NumN i = 1; i <= n; i++) {
            y(i) = x(i);
        }
    }

    template<typename T>
    void math21_operator_vec_to_set(const Tensor<T> &x, _Set<T> &S) {
        NumN n = x.size();
        S.clear();
        for (NumN i = 1; i <= n; i++) {
            S.add(x(i));
        }
    }

    template<typename T>
    void math21_operator_set_to_vec(const _Set<T> &S, Tensor<T> &x) {
        NumN n = S.size();
        x.setSize(n);
        for (NumN i = 1; i <= n; i++) {
            x(i) = S(i);
        }
    }

    template<typename T>
    void math21_operator_vector_find_subvector(const Tensor<T> &v, const Tensor<T> &s, VecN &where) {
        NumN ns = s.size();
        NumN n = v.size();
        Tensor<T> x;
        SeqceN index;
        NumN i = 1;
        while (i <= n) {
            if (i - 1 + ns > n)break;
            math21_operator_share_vector_part_using_from_to(v, x, i, i - 1 + ns);
            if (!math21_op_isEqual(x, s)) {
                ++i;
            } else {
                index.push(i);
                i = i + ns;
            }
        }
        where.setSize(index.size());
        math21_operator_container_set(index, where);
    }

    template<typename T>
    void math21_operator_vector_replace(const Tensor<T> &x, Tensor<T> &y, const Tensor<T> &s, const Tensor<T> &t) {
        VecN where;
        math21_operator_vector_find_subvector(x, s, where);
        NumN n = where.size();
        NumN ns = s.size();
        NumN nt = t.size();
        NumN ny = x.size() - n * ns + n * nt;
        y.setSize(ny);
        VecN ks(n + 1);
        math21_op_vector_set_by_vector(where, ks);
        ks(n + 1) = x.size() + 1;
        n = ks.size();
        for (NumN i = 1; i <= n; ++i) {
            NumN offset_x, offset_1, offset_2 = 0, n1, n2;
            NumN k = ks(i);
            if (i == 1) {
                offset_x = 0;
                offset_1 = 0;
                offset_2 = k - 1;
                n1 = k - 1;
            } else {
                NumN k_pre = ks(i - 1);
                offset_x = k_pre - 1 + ns;
                offset_1 = offset_x - (i - 1) * ns + (i - 1) * nt;
                offset_2 = k - 1 - (i - 1) * ns + (i - 1) * nt;
                n1 = k - (k_pre + ns);
            }
            n2 = nt;
            if (n1 > 0) {
                math21_op_vector_set_by_vector(x, y, 1, 1, offset_x, offset_1, n1);
            }
            if (i < n) {
                if (n2 > 0) {
                    math21_op_vector_set_by_vector(t, y, 1, 1, 0, offset_2, n2);
                }
            }
        }
    }

}