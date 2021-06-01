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

#include <limits>
#include "inner.h"

namespace math21 {

    /////////////////
    template<typename VecType>
    NumB math21_operator_container_isfinite(const VecType &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
        for (i = 1; i <= n; ++i) {
            if (!xjisfinite(A(i))) {
                return 0;
            }
        }
        return 1;
    }

    // T is basic type
    template<typename T>
    NumB math21_operator_isfinite(const T &x) {
        return xjisfinite(x);
    }

    // set MATH21_MAX to x with value infinite ignoring sign.
    template<typename T>
    void math21_operator_clip_pos_inf(T &x) {
        if (xjisinf(x)) {
            x = MATH21_MAX;
        }
    }

    // set MATH21_MIN to x with value infinite ignoring sign.
    template<typename T>
    void math21_operator_clip_neg_inf(T &x) {
        if (xjisinf(x)) {
            x = MATH21_MIN;
        }
    }

    // set MATH21_MAX to x with value infinite ignoring sign.
    template<typename VecType>
    void math21_operator_container_clip_pos_inf(VecType &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
        for (i = 1; i <= n; ++i) {
            if (xjisinf(A(i))) {
                A(i) = MATH21_MAX;
            }
        }
    }


    // set MATH21_MIN to x with value infinite ignoring sign.
    template<typename VecType>
    void math21_operator_container_clip_neg_inf(VecType &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
        for (i = 1; i <= n; ++i) {
            if (xjisinf(A(i))) {
                A(i) = MATH21_MIN;
            }
        }
    }

    template<typename T>
    void math21_operator_clip_not_less_than_eps(T &x) {
        if (x < MATH21_EPS) {
            x = MATH21_EPS;
        }
    }

    template<typename VecType>
    void math21_operator_container_clip_not_less_than_eps(VecType &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
        for (i = 1; i <= n; ++i) {
            if (A(i) < MATH21_EPS) {
                A(i) = MATH21_EPS;
            }
        }
    }

    // set MATH21_MAX to x with value infinite ignoring sign.
    template<typename VecType>
    void math21_operator_container_clip_zero_and_pos_inf(VecType &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
        for (i = 1; i <= n; ++i) {
            if (xjisinf(A(i))) {
                A(i) = MATH21_MAX;
            } else if (A(i) == 0) {
                A(i) = MATH21_EPS;
            }
        }
    }

    // a <= A(i) <= b
    template<typename VecType>
    void math21_operator_container_clip_in(VecType &A, NumR a, NumR b) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            if (A(i) > b) {
                A(i) = b;
            } else if (A(i) < a) {
                A(i) = a;
            }
        }
    }

    //////////////////////
    /* RULE: all container ops can not call set size method.
 *
 * */
    template<typename T, template<typename> class Container>
    NumB math21_clip_container(Container<T> &A) {
        NumB clipped = 0;
        NumN i, n;
        n = A.size();
        for (i = 1; i <= n; ++i) {
            T &val = A(i);
            if (val > MATH21_MAX) {
                val = MATH21_MAX;
                if (clipped == 0) {
                    clipped = 1;
                }
            } else if (val < MATH21_MIN) {
                val = MATH21_MIN;
                if (clipped == 0) {
                    clipped = 1;
                }
            }
        }
        return clipped;
    }

    template<typename T, template<typename> class Container>
    NumB math21_check_clip_container(const Container<T> &A) {
        NumB clipped = 0;
        NumN i, n;
        n = A.size();
        for (i = 1; i <= n; ++i) {
            const T &val = A(i);
            if (val > MATH21_MAX) {
                clipped = 1;
                return clipped;
            } else if (val < MATH21_MIN) {
                clipped = 1;
                return clipped;
            }
        }
        return clipped;
    }

    ////////////////////


    template<typename T, typename S>
    NumB math21_operator_num_isEqual(T x, S y, NumR epsilon = 0) {
        NumR tmp;
        tmp = y - x;
        if (xjabs(tmp) > epsilon) {
            return 0;
        }
        return 1;
    }

    template<typename T, typename S>
    NumB math21_operator_num_isLarger(T x, S y, NumR epsilon = 0) {
        NumR tmp;
        tmp = x - y;
        if (tmp > epsilon) {
            return 0;
        }
        return 1;
    }

    template<typename VecType>
    NumN math21_operator_container_number_of_equal(NumR k, const VecType &x, NumN offset = 0) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        NumN sum = 0;
        for (NumN i = offset + 1; i <= n; ++i) {
            if (k == x(i)) {
                sum = sum + 1;
            }
        }
        return sum;
    }

    template<typename VecType1, typename VecType2>
    NumN math21_operator_container_count_element_equal_at_same_pos(const VecType1 &x, const VecType2 &y) {
        NumN n = xjmin(x.size(), y.size());
        NumN count = 0;
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) == y(i)) {
                count = count + 1;
            }
        }
        return count;
    }

    // indexes.size <= xjmin(x.size(), y.size());
    template<typename VecType1, typename VecType2, typename VecType3>
    void _math21_operator_container_arg_equal_or_not(
            const VecType1 &x, const VecType2 &y, VecType3 &indexes, NumB isEqual, NumB isCheckingCommonOnly = 1) {
        NumN n_equal = math21_operator_container_count_element_equal_at_same_pos(x, y);
        NumN n = xjmin(x.size(), y.size());
        NumN n_max = xjmax(x.size(), y.size());
        if (isEqual) {
            if (n_equal == 0) {
                indexes.clear();
                return;
            }
            indexes.setSize(n_equal);
        } else {
            NumN n_not_equal;
            if (isCheckingCommonOnly) {
                n_not_equal = n - n_equal;
            } else {
                n_not_equal = n_max - n_equal;
            }
            if (n_not_equal == 0) {
                indexes.clear();
                return;
            }
            indexes.setSize(n_not_equal);
        }
        NumN j = 1;
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) == y(i)) {
                if (isEqual == 1) {
                    indexes(j) = i;
                    ++j;
                }
            } else {
                if (!isEqual) {
                    indexes(j) = i;
                    ++j;
                }
            }
        }
        if (!isEqual && !isCheckingCommonOnly) {
            for (NumN i = n + 1; i <= n_max; ++i) {
                indexes(j) = i;
                ++j;
            }
        }
    }

    template<typename VecType1, typename VecType2, typename VecType3>
    void math21_operator_container_argequal(
            const VecType1 &x, const VecType2 &y, VecType3 &indexes) {
        _math21_operator_container_arg_equal_or_not(x, y, indexes, 1);
    }

    template<typename VecType1, typename VecType2, typename VecType3>
    void math21_operator_container_arg_not_equal(
            const VecType1 &x, const VecType2 &y, VecType3 &indexes, NumB isCheckingCommonOnly = 1) {
        _math21_operator_container_arg_equal_or_not(x, y, indexes, 0, isCheckingCommonOnly);
    }

    template<typename VecType>
    NumB math21_operator_container_is_positive(const VecType &x, NumR epsilon = MATH21_MIN_POSITIVE_NUMR) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) <= epsilon) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumN math21_operator_container_not_positive_first(const VecType &x, NumR epsilon = MATH21_MIN_POSITIVE_NUMR) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) <= epsilon) {
                return i;
            }
        }
        return 0;
    }

    template<typename VecType>
    NumB math21_operator_container_is_positive_debug(const VecType &x) {
        if (!math21_operator_container_is_positive(x)) {
            NumN i = math21_operator_container_not_positive_first(x);
//            std::cout << "i = " << i
//                      << "\nx(i) = " << x(i);
            return 0;
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_is_larger_number(const VecType &x, NumR k) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) <= k) {
                return 0;
            }
        }
        return 1;
    }

    // xi, yi can not compare directly.
    // Note: comparing VecN and VecZ
    // x(i)>y(i), for any i
    template<typename VecType, typename VecType2>
    NumB math21_operator_container_is_larger(const VecType &x, const VecType2 &y) {
        MATH21_ASSERT(x.size() == y.size())
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) >= 0 && y(i) >= 0) {
                if (x(i) <= y(i)) {
                    return 0;
                }
            } else if (x(i) < 0 && y(i) < 0) {
                if (x(i) <= y(i)) {
                    return 0;
                }
            } else if (x(i) < 0) {
                return 0;
            }
        }
        return 1;
    }

    // xi, yi can not compare directly.
    // Note: comparing VecN and VecZ
    // x(i)>=y(i), for any i
    template<typename VecType, typename VecType2>
    NumB math21_operator_container_is_not_less(const VecType &x, const VecType2 &y) {
        MATH21_ASSERT(x.size() == y.size())
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) >= 0 && y(i) >= 0) {
                if (x(i) < y(i)) {
                    return 0;
                }
            } else if (x(i) < 0 && y(i) < 0) {
                if (x(i) < y(i)) {
                    return 0;
                }
            } else if (x(i) < 0) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_is_less(const VecType &x, NumR k) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) >= k) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_is_not_larger(const VecType &x, NumR k) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
//        NumN sum = 0;
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) > k) {
                return 0;
            }
        }
        return 1;
    }

    template<typename VecType>
    NumB math21_operator_container_is_not_less(const VecType &x, NumR k) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) < k) {
                return 0;
            }
        }
        return 1;
    }


    ////////////////////
    template<typename S, typename VecType,
            template<typename> class Container>
    void math21_operator_container_add(NumR k, const VecType &x, Container<S> &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = (S) (k + x(i));
        }
        MATH21_ASSERT_FINITE(math21_operator_container_isfinite(y))
    }


    template<typename S, template<typename> class Container>
    void math21_operator_container_addTo(NumR k, Container<S> &x) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            x(i) = (S) (k + x(i));
        }
    }

    template<typename T, typename S, template<typename> class Container,
            template<typename> class Container2>
    void math21_operator_container_divide(NumR k, const Container<T> &x, Container2<S> &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        T EPS = std::numeric_limits<T>::epsilon();
        for (NumN i = 1; i <= n; ++i) {
            MATH21_ASSERT(xjabs(x(i)) > EPS, "divide zero!" << x(i));
            y(i) = (S) (k / (x(i)));
        }
    }

    template<typename S, template<typename> class Container>
    void math21_operator_container_divide_to(NumR k, Container<S> &x) {
        NumN n = x.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            MATH21_ASSERT(x(i) > MATH21_EPS, "divide zero!" << x(i));
            x(i) = (S) (k / (x(i)));
        }
    }

    template<typename S, typename VecType,
            template<typename> class Container>
    void math21_operator_container_sqrt(const VecType &x, Container<S> &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = (S) (xjsqrt(x(i)));
        }
    }

    template<typename S, template<typename> class Container>
    void math21_operator_container_sqrt_to(Container<S> &x) {
        MATH21_ASSERT_POSITIVE(math21_operator_container_is_positive_debug(x))
        NumN n = x.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            x(i) = (S) (xjsqrt(x(i)));
        }
    }

    template<typename S, typename VecType,
            template<typename> class Container>
    void math21_operator_container_square(const VecType &x, Container<S> &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = (S) (xjsquare(x(i)));
        }
    }

    template<typename S, template<typename> class Container>
    void math21_operator_container_square_to(Container<S> &x) {
        math21_operator_container_square(x, x);
    }

    template<typename VecType>
    void math21_operator_container_cos(const VecType &A, VecType &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B.at(i) = xjcos(A(i));
        }
    }

    template<typename VecType>
    void math21_operator_container_cosTo(VecType &A) {
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            A.at(i) = xjcos(A(i));
        }
    }

    template<typename VecType>
    void math21_operator_container_sin(const VecType &A, VecType &B) {
        MATH21_ASSERT(A.size() == B.size(), "vector size doesn't match");
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            B.at(i) = xjsin(A(i));
        }
    }

    template<typename VecType1, typename VecType2>
    void math21_operator_container_log_can_onto(const VecType1 &x, VecType2 &y) {
        MATH21_ASSERT_CHECK_VALUE(math21_operator_container_is_larger_number(x, 0), "" << x.log("x"))

        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = xjlog(x(i));
        }
        MATH21_ASSERT_FINITE(math21_operator_container_isfinite(y))
    }

    //must clip, because exp(800) is infinite in computer.
    template<typename VecType1, typename VecType2>
    void math21_operator_container_exp(const VecType1 &x, VecType2 &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = xjexp(x(i));
        }
        math21_operator_container_clip_zero_and_pos_inf(y);
    }

    template<typename VecType1, typename VecType2>
    void math21_operator_container_round(const VecType1 &x, VecType2 &y, NumN m) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            y(i) = xjround(x(i), m);
        }
    }

    template<typename VecType>
    void math21_operator_container_round_to(VecType &x, NumN m) {
        math21_operator_container_round(x, x, m);
    }

    template<typename VecType>
    NumB math21_operator_container_isContinuousIntegers(const VecType &x, NumZ k = 1) {
        NumN n = x.size();
        MATH21_ASSERT(n >= 1);
        for (NumN i = 1; i <= n; ++i) {
            if (x(i) != k) {
                return 0;
            }
            ++k;
        }
        return 1;
    }
}