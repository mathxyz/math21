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

#include "ten.h"
#include "tenView.h"
#include "tenSub.h"

namespace math21 {

    void math21_tool_setSize(Seqce<TenR> &X, const VecN &shape);

    // v is vector.
    template<typename T>
    void math21_tool_vec_2_seqce(const Tensor<T> &v, Seqce<Tensor<T> > &w) {
        MATH21_ASSERT(v.dims() == 1);
        if (w.size()!=v.size()) {
            w.setSize(v.size());
        }
        VecN d(1);
        d = 1;
        math21_tool_setSize(w, d);
        for (NumN i = 1; i <= w.size(); i++) {
            TenR &x = w.at(i);
            x(1) = v(i);
        }
    }

    template<typename T>
    NumB math21_operator_seqce_isEqual(Seqce<Tensor<T> > &v, Seqce<Tensor<T> > &w) {
        MATH21_ASSERT(v.size() == w.size());
        for (NumN i = 1; i <= w.size(); i++) {
            if (!math21_operator_isEqual(v.at(i), w.at(i))) {
                return 0;
            }
        }
        return 1;
    }

    // m is standard TenR type.
    template<typename T>
    NumB math21_tensor_is_standard_TenR(const Tensor<T> &m) {
        if (typeid(T) == typeid(NumR) && m.isStandard()) {
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T>
    void math21_convert_tensor_to_standard_TenR(const Tensor<T> &A, TenR &B) {
        MATH21_ASSERT(!math21_tensor_is_standard_TenR(A))
        B.setSize(A.shape());
        math21_operator_tensor_assign_elementwise(B, A);
    }

    template<typename T>
    const Tensor<T> &math21_tool_choose_non_empty(const Tensor<T> &A, const Tensor<T> &B) {
        if (!A.isEmpty()) {
            return A;
        } else {
            return B;
        }
    }

    template<typename T>
    Tensor<T> &math21_tool_choose_non_empty(Tensor<T> &A, Tensor<T> &B) {
        if (!A.isEmpty()) {
            return A;
        } else {
            return B;
        }
    }

}