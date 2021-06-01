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
    template<typename VecType1, typename VecType2>
    void math21_convert_container_to_container(const VecType1 &A, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size())
        NumN n = A.size();
        for (NumN i = 1; i <= n; ++i) {
            B(i) = A(i);
        }
    }

    template<typename T, template<typename> class Container,
            typename VecType_std>
    void math21_convert_container_2_std_vec(const Container<T> &A, VecType_std &B) {
        MATH21_ASSERT(A.size() == B.size())
        NumN n = A.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            B[i - 1] = A(i);
        }
    }

    template<typename T, template<typename> class Container,
            typename VecType_std>
    void math21_convert_std_vec_2_container(const VecType_std &A, Container<T> &B) {
        MATH21_ASSERT(A.size() == B.size())
        NumN n = A.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            B(i) = A[i - 1];
        }
    }

    template<typename T>
    void math21_c_array_set(NumN n, const T *A, T *B) {
        NumN i;
#pragma omp parallel for
        for (i = 0; i < n; ++i) {
            B[i] = A[i];
        }
    }

    // prone to misuse
    template<typename T, typename T2>
    void math21_c_array_convert(NumN n, const T *A, T2 *B) {
        NumN i;
#pragma omp parallel for
        for (i = 0; i < n; ++i) {
            B[i] = (T2)A[i];
        }
    }

    template<typename StdVecType1,
            typename StdVecType2>
    void math21_convert_stdvec_2_stdvec(const StdVecType1 &A, StdVecType2 &B) {
        MATH21_ASSERT(A.size() == B.size())
        NumN n = A.size();
//#pragma omp parallel for
        for (NumN i = 0; i < n; ++i) {
            B[i] = A[i];
        }
    }

    template<typename StdVecType>
    void math21_tool_copy_stdvec_2_stdvec(const StdVecType &A, StdVecType &B) {
        NumN n = A.size();
        if (n == 0) {
            return;
        }
        if (B.size() != n) {
            B.resize(n);
        }
//#pragma omp parallel for
        for (NumN i = 0; i < n; ++i) {
            B[i] = A[i];
        }
    }
}