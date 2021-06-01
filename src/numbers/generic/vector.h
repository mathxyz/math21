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
#include "arithmetic.h"

namespace math21 {
    // B=A+k
    template<typename VecType1, typename VecType2>
    void math21_operator_container_add_A_k(const VecType1 &A, NumR k, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = A(i) + k;
        }
    }

    // B=A-k
    template<typename VecType1, typename T, template<typename> class Container>
    void math21_operator_container_subtract_A_k(const VecType1 &A, NumR k, Container<T> &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = static_cast<T>(A(i) - k);
        }
    }

    // A=A-k
    template<typename VecType>
    void math21_operator_container_subtract_A_k_to(VecType &A, NumR k) {
        math21_operator_container_subtract_A_k(A, k, A);
    }

    // B=k-A
    template<typename VecType1, typename VecType2>
    void math21_operator_container_subtract_k_A(NumR k, const VecType1 &A, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = k - A(i);
        }
    }

    // B=k-A
    template<typename VecType>
    void math21_operator_container_subtract_k_A_to(NumR k, VecType &A) {
        math21_operator_container_subtract_k_A(k, A, A);
    }

    // C=A-B
    template<typename VecType1, typename VecType2, typename VecType3>
    void math21_operator_container_subtract_to_C(const VecType1 &A, const VecType2 &B, VecType3 &C) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        MATH21_ASSERT(A.size() == C.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            C(i) = A(i) - B(i);
        }
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_container_isfinite(C))
    }

    // A=A-B
    template<typename VecType, typename VecType2>
    void math21_operator_container_subtract_to_A(VecType &A, const VecType2 &B) {
        math21_operator_container_subtract_to_C(A, B, A);
    }

    // B=A-B
    template<typename VecType, typename VecType2>
    void math21_operator_container_subtract_to_B(const VecType &A, VecType2 &B) {
        math21_operator_container_subtract_to_C(A, B, B);
    }

    template<typename VecType1, typename VecType2>
    void math21_operator_container_degree2radian(const VecType1 &A, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = xjdegree2radian(A(i));
        }
    }

    template<typename VecType1>
    void math21_operator_container_degree2radian_to(VecType1 &A) {
        math21_operator_container_degree2radian(A, A);
    }

    template<typename VecType1, typename VecType2>
    void math21_operator_container_radian2degree(const VecType1 &A, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = xjradian2degree(A(i));
        }
    }

    template<typename VecType1>
    void math21_operator_container_radian2degree_to(VecType1 &A) {
        math21_operator_container_radian2degree(A, A);
    }

    // B = power(A, p)
    template<typename VecType1, typename VecType2>
    void math21_operator_container_power(const VecType1 &A, VecType2 &B, NumR p) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = xjpow(A(i), p);
        }
    }

    // B = power(A, p)
    template<typename VecType1>
    void math21_operator_container_power_to(VecType1 &A, NumR p) {
        math21_operator_container_power(A, A, p);
    }

    // C=A/B
    template<typename VecType1, typename VecType2, typename VecType3>
    void math21_operator_container_divide_to_C(const VecType1 &A, const VecType2 &B, VecType3 &C) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        MATH21_ASSERT(A.size() == C.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            MATH21_ASSERT(xjabs(B(i)) > MATH21_EPS, "divide zero!" << B(i));
            C(i) = A(i) / B(i);
        }
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_container_isfinite(C))
    }

    // A=A/B
    template<typename VecType, typename VecType2>
    void math21_operator_container_divide_to_A(VecType &A, const VecType2 &B) {
        math21_operator_container_divide_to_C(A, B, A);
    }

    // B=A/B
    template<typename VecType, typename VecType2>
    void math21_operator_container_divide_to_B(const VecType &A, VecType2 &B) {
        math21_operator_container_divide_to_C(A, B, B);
    }

    // C=A+B
    template<typename VecType1, typename VecType2, typename VecType3>
    void math21_operator_container_addToC(const VecType1 &A, const VecType2 &B, VecType3 &C) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        MATH21_ASSERT(A.size() == C.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            C(i) = A(i) + B(i);
        }
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_container_isfinite(C))
    }

    // A=A+B
    template<typename VecType, typename VecType2>
    void math21_operator_container_addToA(VecType &A, const VecType2 &B) {
        math21_operator_container_addToC(A, B, A);
    }

    // B=A+B
    template<typename VecType, typename VecType2>
    void math21_operator_container_addToB(const VecType &A, VecType2 &B) {
        math21_operator_container_addToC(A, B, B);
    }

    // user should make sure that A(i) doesn't have type NumN when k<0.
    template<typename VecType>
    void math21_operator_container_letters(VecType &A, NumZ k = 1, NumN from = 0, NumN to = 0) {
        MATH21_ASSERT(!A.isEmpty());
        NumN i;
        NumN n = A.size();
        if (from == 0) {
            from = 1;
        }
        if (to == 0) {
            to = n;
        }
        MATH21_ASSERT(from >= 1 && from <= to && to <= n)
        for (i = from; i <= to; ++i) {
            A(i) = k;
            ++k;
        }
    }

    // user should make sure that A(i) doesn't have type NumN when k<0.
    template<typename VecType, typename T>
    void math21_operator_container_set_value(VecType &A, T k, T step, NumN from = 0, NumN to = 0) {
        MATH21_ASSERT(!A.isEmpty());
        NumN i;
        NumN n = A.size();
        if (from == 0) {
            from = 1;
        }
        if (to == 0) {
            to = n;
        }
        MATH21_ASSERT(from >= 1 && from <= to && to <= n)
        for (i = from; i <= to; ++i) {
            A(i) = k;
            k = k + step;
        }
    }

    // user should make sure that A(i) doesn't have type NumN when k<0.
    template<typename T, template<typename> class Container>
    void math21_operator_container_set_num(Container<T> &A, const T &k = 1, NumN from = 0, NumN to = 0) {
        MATH21_ASSERT(!A.isEmpty());
        NumN i;
        NumN n = A.size();
        if (from == 0) {
            from = 1;
        }
        if (to == 0) {
            to = n;
        }
        MATH21_ASSERT(from >= 1 && from <= to && to <= n)
        for (i = from; i <= to; ++i) {
            A(i) = k;
        }
    }

    // C=A*B
    template<typename VecType>
    void math21_operator_container_SchurProduct(const VecType &A, const VecType &B, VecType &C) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        MATH21_ASSERT(A.size() == C.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            C.at(i) = A(i) * B(i);
        }
    }

    // A=A*B
    template<typename VecType>
    void math21_operator_container_SchurProduct_to_A(VecType &A, const VecType &B) {
        math21_operator_container_SchurProduct(A, B, A);
    }

    // B=A*B
    template<typename VecType>
    void math21_operator_container_SchurProduct_to_B(const VecType &A, VecType &B) {
        math21_operator_container_SchurProduct(A, B, B);
    }

    // return first min value.
    template<typename T, template<typename> class Container>
    T math21_operator_container_min(const Container<T> &m) {
        NumN i;
        NumN n = m.size();
        MATH21_ASSERT(n >= 1);
        NumN k = 1;
        for (i = 2; i <= n; ++i) {
            if (m(i) < m(k)) {
                k = i;
            }
        }
        return m(k);
    }

    template<typename T, template<typename> class Container>
    T math21_operator_container_max(const Container<T> &m) {
        NumN i;
        NumN n = m.size();
        MATH21_ASSERT(n >= 1);
        NumN k = 1;
        for (i = 2; i <= n; ++i) {
            if (m(i) > m(k)) {
                k = i;
            }
        }
        return m(k);
    }

    template<typename VecType>
    NumN math21_operator_container_argmin(const VecType &m) {
        NumN i;
        NumN n = m.size();
        MATH21_ASSERT(n >= 1);
        NumN k = 1;
        for (i = 2; i <= n; ++i) {
            if (m(i) < m(k)) {
                k = i;
            }
        }
        return k;
    }


    template<typename T, template<typename> class Container>
    NumN math21_operator_container_index(const Container<T> &m, const T &x) {
        NumN n = m.size();
        for (NumN i = 1; i <= n; ++i) {
            if (m(i) == x) {
                return i;
            }
        }
        return 0;
    }

    template<typename T, template<typename> class Container>
    NumN math21_operator_container_arg(const Container<T> &m, const T &x) {
        return math21_operator_container_index(m, x);
    }

    // argmax from index k = 1
    template<typename VecType>
    NumN math21_operator_container_argmax(const VecType &m, NumN k = 1) {
        NumN i;
        NumN n = m.size();
        MATH21_ASSERT(k >= 1 && k <= n);
        for (i = k + 1; i <= n; ++i) {
            if (m(i) > m(k)) {
                k = i;
            }
        }
        return k;
    }

    //argmax every element of v.
    template<template<typename> class Container, typename VecType1, typename VecType2>
    void math21_operator_container_argmax(const Container<VecType1> &v, VecType2 &m) {
        MATH21_ASSERT(v.size() > 0, "v is empty");
        MATH21_ASSERT(m.size() == v.size());
        for (NumN i = 1; i <= v.size(); ++i) {
            m(i) = math21_operator_container_argmax(v(i));
        }
    }

    template<typename T, template<typename> class Container>
    T math21_operator_container_multiply_some(const Container<T> &x, NumN n, NumN offset = 0) {
        if (n == 0) {
            return 0;
        }
        MATH21_ASSERT(offset + n <= x.size());
        T sum = 1;
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            sum = sum * x(offset + i);
        }
        return sum;
    }

    template<typename T, template<typename> class Container>
    T math21_operator_container_multiply_all(const Container<T> &x) {
        return math21_operator_container_multiply_some(x, x.size());
    }

    template<typename VecType>
    NumB math21_operator_container_isEqual(const VecType &x, const VecType &y, NumR epsilon = 0) {
        if (x.size() != y.size()) {
            return 0;
        }
        NumN n = x.size();
        if (n == 0) {
            return 1;
        }
        for (NumN i = 1; i <= n; ++i) {
            if (!math21_point_isEqual(x(i), y(i), epsilon)) {
                return 0;
            }
        }
        return 1;
    }

    template<typename T, typename VecType>
    NumB math21_operator_container_isEqual_c_array(const VecType &x, const T *y, NumR epsilon = 0) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        NumR tmp;
        for (NumN i = 1; i <= n; ++i) {
            tmp = y[i - 1] - x(i);
            if (xjabs(tmp) > epsilon) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_isEqual_num(const VecType &x, NumR k) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) != k) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_isEqualZero(const VecType &x) {
        return math21_operator_container_isEqual_num(x, 0);
    }

    template<typename VecType>
    NumB math21_operator_check_container_is_nan(const VecType &x) {
        NumN n = x.size();
        for (NumN i = 1; i <= n; ++i) {
            if (x(i)!=x(i)) {
                return 1;
            }
        }
        return 0;
    }

    template<typename VecType>
    NumR math21_operator_container_sum(const VecType &A, NumN n) {
        NumN i;
        NumR sum = 0;

        NumN size = A.size();
        if (n == 1) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += A(i);
        } else if (n == 2) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += xjsquare(A(i));
            sum = xjsqrt(sum);
        } else {
            MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
        }
        MATH21_ASSERT_FINITE(math21_operator_isfinite(sum))
        return sum;
    }

    template<typename VecType>
    NumR math21_operator_container_mean(const VecType &A) {
        MATH21_ASSERT(!A.isEmpty());
        NumR sum = math21_operator_container_sum(A, 1);
        return sum / A.size();
    }

    template<typename VecType>
    NumR math21_operator_container_norm(const VecType &A, NumN n) {
        NumN i;
        NumR sum = 0;
        NumN size = A.size();
        if (n == 1) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += xjabs(A(i));
        } else if (n == 2) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += xjsquare(A(i));
            sum = xjsqrt(sum);
        } else {
            MATH21_ASSERT(0, "norm other than 1, 2 not supported currently");
        }
        MATH21_ASSERT_FINITE(math21_operator_isfinite(sum))
        return sum;
    }

    template<typename VecType1, typename VecType2>
    NumR math21_operator_container_distance(const VecType1 &A, const VecType2 &B, NumR n) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        MATH21_ASSERT(n > 0);
        NumN i;
        NumR sum = 0;
        NumN size = A.size();
        if (n == 1) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += xjabs(A(i) - B(i));
        } else if (n == 2) {
//#pragma omp parallel for
            for (i = 1; i <= size; ++i) sum += xjsquare(A(i) - B(i));
            sum = xjsqrt(sum);
        } else {
            for (i = 1; i <= size; ++i) sum += xjpow(xjabs(A(i) - B(i)), n);
            sum = xjpow(sum, 1 / n);
        }
        MATH21_ASSERT_FINITE(math21_operator_isfinite(sum))
        return sum;
    }

    template<typename VecType>
    NumR math21_operator_container_InnerProduct(NumR k, const VecType &A, const VecType &B) {
        MATH21_ASSERT(A.size() == B.size());
        if (k == 0) {
            return 0;
        }
        NumN i;
        NumN n = A.size();
        NumR y = 0;
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            y = y + (A(i) * B(i));
        }
        y = y * k;
        MATH21_ASSERT_FINITE(math21_operator_isfinite(y))
        return y;
    }

    template<typename VecType>
    void math21_operator_container_CrossProduct(const VecType &A, const VecType &B, VecType &C) {
        MATH21_ASSERT(A.size() == 3);
        MATH21_ASSERT(A.size() == B.size());
        if (C.size() != 3) {
            C.setSize(3);
        }
        C(1) = A(2) * B(3) - A(3) * B(2);
        C(2) = A(3) * B(1) - A(1) * B(3);
        C(3) = A(1) * B(2) - A(2) * B(1);
    }

    // y = k*x + b
    template<typename VecType, typename VecType2>
    void math21_operator_container_linear_kx_b(NumR k, const VecType &x, NumR b, VecType2 &y) {
        MATH21_ASSERT(!x.isEmpty(), "empty matrix");
        MATH21_ASSERT(x.size() == y.size());
        NumN i;
        NumN n = x.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            y(i) = k * x(i) + b;
        }
    }

    // x <- k*x + b
    template<typename VecType>
    void math21_operator_container_linear_kx_b_to(NumR k, VecType &x, NumR b) {
        math21_operator_container_linear_kx_b(k, x, b, x);
    }

    // C = k1*A
    template<typename VecType, typename T, template<typename> class Container>
    void math21_operator_container_linear(NumR k1, const VecType &A, Container<T> &C) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        MATH21_ASSERT(A.size() == C.size());
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            C(i) = static_cast<T>(k1 * A(i));
        }
    }

    // B = A/A.norm(n)
    template<typename VecType, typename VecType2>
    void math21_operator_container_normalize_to_B(const VecType &A, VecType2 &B, NumN n) {
        NumR k = math21_operator_container_norm(A, n);
        MATH21_ASSERT(!math21_operator_num_isEqual(k, 0));
        math21_operator_container_linear(1 / k, A, B);
    }

    // A = A/A.norm(n)
    template<typename VecType>
    void math21_operator_container_normalize_to_A(VecType &A, NumN n) {
        math21_operator_container_normalize_to_B(A, A, n);
    }

    // A = k1*A
    template<typename VecType>
    void math21_operator_container_linear_to_A(NumR k1, VecType &A) {
        math21_operator_container_linear(k1, A, A);
    }

    // C = k1*A + k2*B
    template<typename VecType, typename VecType2, typename T, template<typename> class Container>
    void math21_operator_container_linear(NumR k1, const VecType &A, NumR k2, const VecType2 &B, Container<T> &C) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        MATH21_ASSERT(A.size() == B.size());
        MATH21_ASSERT(A.size() == C.size());

        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            C(i) = static_cast<T>(k1 * A(i) + k2 * B(i));
        }
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_container_isfinite(C))
    }

    // E = k1*A + k2*B + k3*C + k4*D
    template<typename VecType, typename VecType2,
            typename VecType3, typename VecType4, typename VecType5>
    void math21_operator_container_linear_to_E(
            NumR k1, const VecType &A,
            NumR k2, const VecType2 &B,
            NumR k3, const VecType3 &C,
            NumR k4, const VecType4 &D,
            VecType5 &E) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        MATH21_ASSERT(A.size() == B.size());
        MATH21_ASSERT(A.size() == C.size());
        MATH21_ASSERT(A.size() == D.size());
        MATH21_ASSERT(A.size() == E.size());

        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            E(i) = k1 * A(i) + k2 * B(i) + k3 * C(i) + k4 * D(i);
        }
        MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_container_isfinite(E))
    }

    // A = k1*A + k2*B + k3*C + k4*D
    template<typename VecType, typename VecType2,
            typename VecType3, typename VecType4>
    void math21_operator_container_linear_to_A(
            NumR k1, VecType &A,
            NumR k2, const VecType2 &B,
            NumR k3, const VecType3 &C,
            NumR k4, const VecType4 &D) {
        math21_operator_container_linear_to_E(k1, A, k2, B, k3, C, k4, D, A);
    }

    // A = k1*A + k2*B
    template<typename VecType, typename VecType2>
    void math21_operator_container_linear_to_A(NumR k1, VecType &A, const NumR k2, const VecType2 &B) {
        math21_operator_container_linear(k1, A, k2, B, A);
    }

    // B = k1*A + k2*B
    template<typename VecType, typename VecType2>
    void math21_operator_container_linear_to_B(NumR k1, const VecType &A, NumR k2, VecType2 &B) {
        math21_operator_container_linear(k1, A, k2, B, B);
    }

    // B=|A|
    template<typename VecType1, typename VecType2>
    void math21_operator_container_abs(const VecType1 &A, VecType2 &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B(i) = xjabs(A(i));
        }
    }

    // B=|A|
    template<typename VecType1>
    void math21_operator_container_abs_to(VecType1 &A) {
        math21_operator_container_abs(A, A);
    }

    template<typename VecType, typename VecType2>
    void math21_operator_container_f_elementwise_unary(const VecType &x, VecType2 &y, NumR (*f)(const NumR &x)) {
        MATH21_ASSERT(!x.isEmpty(), "empty matrix");
        MATH21_ASSERT(x.size() == y.size());
        NumN i;
        NumN n = x.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            y(i) = f(x(i));
        }
    }

    template<typename VecType1, typename VecType2, typename VecType3, typename VecType4>
    void math21_operator_container_f_elementwise_binary(const VecType1 &x1, const VecType2 &x2, VecType3 &y,
                                                        NumR (*f)(const NumR &x1, const NumR &x2),
                                                        const VecType4 &mask) {
        MATH21_ASSERT(!x1.isEmpty(), "empty matrix");
        MATH21_ASSERT(x1.size() == x2.size());
        MATH21_ASSERT(x1.size() == y.size());
        NumN i;
        NumN n = x1.size();
        NumB isUseMask = 0;
        if (mask.size() == y.size()) {
            isUseMask = 1;
        }
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            if (isUseMask) {
                if (mask(i)) {
                    y(i) = f(x1(i), x2(i));
                }
            } else {
                y(i) = f(x1(i), x2(i));
            }
        }
    }

    template<typename VecType1, typename VecType2, typename VecType3, typename VecType>
    void math21_operator_container_f_elementwise_ternary(const VecType1 &x1, const VecType2 &x2, const VecType3 &x3,
                                                         VecType &y,
                                                         NumR (*f)(const NumR &x1, const NumR &x2, const NumR &x3)) {
        MATH21_ASSERT(!x1.isEmpty(), "empty matrix");
        MATH21_ASSERT(x1.size() == x2.size());
        MATH21_ASSERT(x1.size() == x3.size());
        MATH21_ASSERT(x1.size() == y.size());
        NumN i;
        NumN n = x1.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            y(i) = f(x1(i), x2(i), x3(i));
        }
    }

    template<typename VecType1, typename VecType2>
    void math21_operator_container_cdf_like(const VecType1 &x, VecType2 &y, NumB startZero) {
        MATH21_ASSERT(y.size() == x.size());
        NumN n = x.size();
        if (startZero) {
            for (NumN i = 1; i <= n; ++i) {
                if (i == 1) {
                    y(i) = 0;
                } else {
                    y(i) = y(i - 1) + x(i - 1);
                }
            }
        } else {
            for (NumN i = 1; i <= n; ++i) {
                if (i == 1) {
                    y(i) = x(i);
                } else {
                    y(i) = y(i - 1) + x(i);
                }
            }
        }
    }

    // collapse
    template<typename T, template<typename> class Container, typename VecType>
    void math21_operator_container_dn_to_d3(const Container<T> &d_x, NumN axis_d2, VecType &d){
        d.setSize(3);
        NumN p1;
        p1 = axis_d2;
        d(1) = math21_operator_container_multiply_some(d_x, p1 - 1, 0);
        d(2) = d_x(p1);
        d(3) = math21_operator_container_multiply_some(d_x, d_x.size() - p1, p1);
        for (NumN i = 1; i <= d.size(); ++i) {
            if (d(i) == 0) {
                d(i) = 1;
            }
        }
    }

    // collapse
    template<typename T, template<typename> class Container, typename VecType>
    void math21_operator_container_dn_to_d5_fix_24(const Container<T> &d_x, NumN axis_d2, NumN axis_d4, VecType &d){
        d.setSize(5);
        NumN p1, p2;
        p1 = axis_d2;
        p2 = axis_d4;
        MATH21_ASSERT(p1 < p2, "collapse axes");
        d(1) = math21_operator_container_multiply_some(d_x, p1 - 1, 0);
        d(2) = d_x(p1);
        d(3) = math21_operator_container_multiply_some(d_x, p2 - p1 - 1, p1);
        d(4) = d_x(p2);
        d(5) = math21_operator_container_multiply_some(d_x, d_x.size() - p2, p2);
        for (NumN i = 1; i <= d.size(); ++i) {
            if (d(i) == 0) {
                d(i) = 1;
            }
        }
    }

}