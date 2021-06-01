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

#include "matrix.h"
#include "ten.h"

namespace math21 {
    // Has error!
    template<typename T>
    const T *math21_memory_tensor_data_address(const Tensor <T> &A) {
        return A.getDataAddress();
    }

    // Has error!
    // But still using and testing!
    template<typename T>
    T *math21_memory_tensor_data_address(Tensor <T> &A) {
        return A.getDataAddress();
    }

    template<typename T>
    void math21_memory_tensor_data_copy_to_tensor(Tensor <T> &A, const T *data) {
        MATH21_ASSERT(A.isContinuous())
        SpaceParas paras = A.getSpace();
        if (data != (const T *) paras.start) {
            math21_memory_memcpy(paras.start, data, sizeof(T) * A.size());
        }
    }

    template<typename T>
    void math21_memory_tensor_data_copy_to_buffer(const Tensor <T> &A, T *data) {
        MATH21_ASSERT(A.isContinuous())
        SpaceParas paras = A.getSpace();
        if (data != (const T *) paras.start) {
            math21_memory_memcpy(data, paras.start, sizeof(T) * A.size());
        }
    }

    // start from k.
    template<typename T>
    NumB math21_operator_isContinuousIntegers(const Tensor <T> &x, NumZ k = 1) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_isContinuousIntegers(x, k);
    }

    template<typename T>
    NumB math21_operator_isEqual(const Tensor <T> &x, const Tensor <T> &y, NumR epsilon = 0) {
        if (y.isSameSize(x.shape()) == 0) {
            return 0;
        }
        if (x.isEmpty()) {
            return 1;
        }
        return math21_operator_container_isEqual(x, y, epsilon);
    }

    template<typename T>
    NumN math21_operator_number_of_equal(NumR k, const Tensor <T> &x) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_number_of_equal(k, x);
    }

    template<typename T>
    NumB math21_operator_is_not_less(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_not_less(x, k);
    }

    template<typename T>
    NumB math21_operator_is_less(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_less(x, k);
    }

    template<typename T>
    NumB math21_operator_is_not_larger(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_not_larger(x, k);
    }

    template<typename T>
    NumB math21_operator_is_larger(const Tensor <T> &x, NumR k) {
        MATH21_ASSERT(!x.isEmpty());
        return math21_operator_container_is_larger_number(x, k);
    }

    template<typename T>
    T math21_operator_multiply_all(const Tensor <T> &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        return math21_operator_container_multiply_all(A);
    }

    // Todo: Deep question: Is it safe if v and w have a piece of common space? (next version)
    template<typename T>
    void math21_operator_share_reshape(const Tensor <T> &v, Tensor <T> &w, const VecN &d) {
        MATH21_ASSERT(v.volume() == math21_operator_multiply_all(d));
        SpaceParas paras = v.getSpace();
        w.setSize(d, &paras);
    }

    // share data
    template<typename T>
    void math21_operator_tensor_shallow_copy(const Tensor <T> &v, Tensor <T> &w) {
        MATH21_ASSERT(v.isContinuous())
        SpaceParas paras = v.getSpace();
        VecN d;
        v.shape(d);
        w.setSize(d, &paras);
    }

    template<typename T>
    void math21_operator_tensor_set_data(Tensor <T> &A, const VecN &d, const T *data, NumB shallow=1) {
        MATH21_ASSERT(A.isBasicType());
        if(shallow){
            A.setSizeNoSpace(d);
            A.setSpace(data, A.size() * sizeof(T));
        }else{
            A.setSize(d);
            math21_memory_tensor_data_copy_to_tensor(A, data);
        }
    }

    template<typename T>
    void math21_operator_tensor_set_data(Tensor <T> &A, NumN n, const T *data, NumB shallow=1) {
        VecN d(1);
        d = n;
        math21_operator_tensor_set_data(A, d, data, shallow);
    }

    template<typename T>
    void math21_operator_share_reshape_to_vector(const Tensor <T> &v, Tensor <T> &w) {
        VecN d(1);
        d = v.volume();
        math21_operator_share_reshape(v, w, d);
    }

    template<typename T>
    void math21_operator_share_reshape_2d_to_3d(const Tensor <T> &v, Tensor <T> &w) {
        MATH21_ASSERT(v.dims() == 2)
        VecN d(3);
        d = 1, v.dim(1), v.dim(2);
        math21_operator_share_reshape(v, w, d);
    }

    template<typename T>
    void math21_operator_share_reshape_remove_dim_1(const Tensor <T> &v, Seqce <Tensor<T>> &ws) {
        m21log("not test", __FUNCTION__);
        MATH21_ASSERT(v.dims() >= 2, "v.dims() = " << v.dims())
        NumN n = v.dim(1);
        VecN d(v.dims() - 1);
        math21_operator_container_set_partially(v.shape(), d, 1);
        NumN volume = math21_operator_container_multiply_all(d);

        ws.setSize(n);
        SpaceParas paras = v.getSpace();

        NumN offset = 0;
        SpaceParas paras_dst;
        for (NumN i = 1; i <= n; ++i) {
            math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
            ws(i).setSize(d, &paras_dst);
            offset = offset + volume;
        }
    }

    // v is seen as vector
    template<typename T>
    void math21_operator_share_part_using_offset(const Tensor <T> &v, NumN offset, const VecN &d, Tensor <T> &w) {
        NumN volume_all = v.size();
        NumN volume = math21_operator_container_multiply_all(d);
        MATH21_ASSERT(offset + volume <= volume_all)
        SpaceParas paras = v.getSpace();
        SpaceParas paras_dst;
        math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
        w.setSize(d, &paras_dst);
    }

    // v is seen as vector
    template<typename T>
    void math21_operator_share_vector_part_using_from_to(const Tensor <T> &v, Tensor <T> &w, NumZ from, NumZ to) {
        from = math21_number_container_pos_check(v.size(), from);
        to = math21_number_container_pos_check(v.size(), to);
        NumN offset = from - 1;
        VecN d(1);
        d = to - offset;
        math21_operator_share_part_using_offset(v, offset, d, w);
    }

    // todo: remove, use below
    // v is seen as vector
    // get i-th part from the whole space, each part having shape d.
    template<typename T>
    void math21_operator_share_part_i_tensor_bak_zz(const Tensor <T> &v, NumN i, const VecN &d, Tensor <T> &w) {
        NumN volume_all = v.size();
        NumN volume = math21_operator_container_multiply_all(d);
        MATH21_ASSERT(i > 0 && i * volume <= volume_all)
        NumN offset = (i - 1) * volume;
        SpaceParas paras = v.getSpace();
        SpaceParas paras_dst;
        math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
        w.setSize(d, &paras_dst);
    }

    // v is seen as vector
    // get i-th part from the whole space, each part having shape d.
    template<typename T>
    void math21_operator_share_part_i_tensor(const Tensor <T> &v, NumN i, const VecN &d, Tensor <T> &w) {
        NumN volume = math21_operator_container_multiply_all(d);
        NumN offset = (i - 1) * volume;
        math21_operator_share_part_using_offset(v, offset, d, w);
    }

    // get i-th part from the whole space, each part having shape (nr,nc).
    template<typename T>
    void math21_operator_share_part_i_mat(const Tensor <T> &v, NumN i, NumN nr, NumN nc, Tensor <T> &w) {
        VecN d(2);
        d = nr, nc;
        math21_operator_share_part_i_tensor(v, i, d, w);
    }

    template<typename T>
    void math21_operator_share_reshape_mat_2_mats(const Tensor <T> &v, Seqce <Tensor<T>> &ws, NumN nr, NumN nc) {
        MATH21_ASSERT(v.dims() == 2, "v.dims() = " << v.dims())
        NumN n = v.dim(1);
        VecN d(v.dims() - 1);
        math21_operator_container_set_partially(v.shape(), d, 1);
        NumN volume = math21_operator_container_multiply_all(d);
        MATH21_ASSERT(volume == nr * nc, "volume = " << volume << ", nr =  " << nr << ", nc = " << nc);
        VecN shape(2);
        shape = nr, nc;

        ws.setSize(n);
        SpaceParas paras = v.getSpace();

        NumN offset = 0;
        SpaceParas paras_dst;
        for (NumN i = 1; i <= n; ++i) {
            math21_memory_getSpace(paras, paras_dst, offset, volume, sizeof(T));
            ws(i).setSize(shape, &paras_dst);
            offset = offset + volume;
        }
    }

    template<typename T>
    void math21_operator_share_reshape(const Tensor <T> &v, Tensor <T> &w) {
        MATH21_ASSERT(v.volume() == w.volume());
        SpaceParas paras = v.getSpace();
        w.setSpace(paras);
    }

    // return shrinked shape
    // e.x., b = 2, 1, 0.
    template<typename T>
    void math21_operator_tensor_shrink_shape(const Tensor <T> &A, const VecN &b, VecN &d) {
        NumN n = math21_operator_number_of_equal(0, b);
        if (n == 0) {
            d.clear();
            return;
        }
        d.setSize(n);
        for (NumN i = 1, j = 1; i <= b.size(); ++i) {
            if (b(i) == 0) {
                d(j) = A.dim(i);
                ++j;
            }
        }
    }

    template<typename T>
    void math21_operator_tensor_shrink_shape_using_axes_with_dim_kept(const Tensor <T> &A, const VecN &axes, VecN &d) {
        if (axes.isEmpty()) {
            d.setSize(A.dims());
            d = 1;
        } else {
            A.shape(d);
            for (NumN i = 1; i <= axes.size(); ++i) {
                math21_tool_assert(axes(i) <= d.size());
                d(axes(i)) = 1;
            }
        }
    }
}