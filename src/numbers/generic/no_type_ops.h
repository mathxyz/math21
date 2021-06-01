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

    template<typename T, typename S, template<typename> class Container1,
            template<typename> class Container2>
    void math21_operator_container_set(const Container1<T> &A, Container2<S> &B) {
        MATH21_ASSERT(A.size() == B.size())
        NumN n = A.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            B.at(i) = A(i);
        }
    }

    // Set B using values of A. offset in [0, n).
    template<typename T, typename S, template<typename> class Container1,
            template<typename> class Container2>
    void math21_operator_container_set_partially(const Container1<T> &A, Container2<S> &B,
                                                 NumN offset1 = 0, NumN offset2 = 0, NumN count = 0) {
        MATH21_ASSERT(offset1 < A.size() && offset2 < B.size());
        if (count == 0) {
            count = xjmin(A.size() - offset1, B.size() - offset2);
        } else {
            MATH21_ASSERT(count + offset1 <= A.size());
            MATH21_ASSERT(count + offset2 <= B.size());
        }
        NumN k;
        for (k = 1; k <= count; ++k) {
            B(offset2 + k) = A(offset1 + k);
        }
    }

    template<typename T, template<typename> class Container>
    void math21_operator_container_swap(Container<T> &x, NumZ pos, NumZ pos2) {
        NumN p1 = math21_number_container_pos_check(x.size(), pos);
        NumN p2 = math21_number_container_pos_check(x.size(), pos2);
        m21_swap(x(p1), x(p2));
    }

    //////
    template<typename VecType>
    void math21_operator_container_reverse(const VecType &x, VecType &y) {
        NumN n = x.size();
        MATH21_ASSERT(y.size() == n);
        for (NumN i = 1, j = n; i <= n; ++i, --j) {
            y(i) = x(j);
        }
    }

    template<typename VecType>
    void math21_operator_reverse(VecType &x) {
        NumN n = x.size();
        NumN n2 = n / 2;
//#pragma omp parallel for
        for (NumN k = 1; k <= n2; ++k) {
            m21_swap(x.at(k), x.at(n + 1 - k));
        }
    }

    // see math21_op_vector_concatenate
    template<typename VecType, typename VecType2, typename VecType3>
    void math21_operator_merge(const VecType &x, const VecType2 &y, VecType3 &z) {
        NumN n1 = x.size();
        NumN n2 = y.size();
        NumN n = n1 + n2;
        z.setSize(n);
//#pragma omp parallel for
        for (NumN k = 1; k <= n; ++k) {
            if (k <= n1) {
                z.at(k) = x(k);
            } else {
                z.at(k) = y(k - n1);
            }
        }
    }

    template<typename VecType>
    void math21_operator_container_sub_from_start(VecType &x, NumN size = 0) {
        MATH21_ASSERT(!x.isEmpty())
        NumN n1 = x.size();
        if (size == 0) {
            x.clear();
            return;
        }
        if (n1 == size) {
            return;
        }
        MATH21_ASSERT(size <= n1)
        NumN n2 = size;
        VecType y;
        y.setSize(n2);
        for (NumN k = 1; k <= n2; ++k) {
            y.at(k) = x(k);
        }
        x.setSize(size);
        x.assign(y);
    }

    // now support negative index
    // 1 <= from <= to <= x.size(), [from , to]
    template<typename VecType>
    void math21_operator_container_subcontainer(const VecType &x, VecType &y, NumZ from, NumZ to = -1) {
        MATH21_ASSERT(!x.isEmpty())
        NumN n1 = x.size();
        from = math21_number_container_pos_check(n1, from);
        to = math21_number_container_pos_check(n1, to);
        MATH21_ASSERT(from >= 1 && from <= to && to <= n1)
        NumN n2 = (NumN) (to + 1 - from);
        if (y.size() != n2) {
            y.setSize(n2);
        }
        NumN offset = (NumN) (from - 1);
        for (NumN k = 1; k <= n2; ++k) {
            y.at(k) = x(offset + k);
        }
    }

    // Todo: change NumN to NumZ
    // u is max, v is number. return 0 if fail.
    template<template<typename> class Container,
            template<typename> class Container2>
    NumB math21_operator_container_increaseNumFromRight(const Container<NumN> &u, Container2<NumN> &v) {
        MATH21_ASSERT(!v.isEmpty());
        MATH21_ASSERT(u.size() == v.size());
        for (NumN i = v.size(); i >= 1; --i) {
            if (v(i) < u(i)) {
                v.at(i) = v(i) + 1;
                return 1;
            } else {
                MATH21_ASSERT(v(i) == u(i), "v(i) = " << v(i) << ", u(i) = " << u(i) << "\n");
                v.at(i) = 1;
            }
        }
        return 0;
    }

    // Todo: change NumN to NumZ
    // u is max, v is number. return 0 if fail.
    // u is end index, start is start index. v is current index.
    template<template<typename> class Container,
            template<typename> class Container2>
    NumB math21_operator_container_increaseNumFromRight(const Container<NumN> &u, Container2<NumN> &v,
                                                        const Container<NumN> &start) {
        MATH21_ASSERT(!v.isEmpty());
        MATH21_ASSERT(u.size() == v.size());
        MATH21_ASSERT(u.size() == start.size());
        for (NumN i = v.size(); i >= 1; --i) {
            if (v(i) < u(i)) {
                v.at(i) = v(i) + 1;
                return 1;
            } else {
                MATH21_ASSERT(v(i) == u(i));
                v.at(i) = start(i);
            }
        }
        return 0;
    }

    // d is shape, a is start index.
    // y-a(n) = sum ((x(i)-a(i))*k(i))
    template<template<typename> class Container,
            typename VecZType, typename VecZType2>
    void math21_operator_number_index_1d_to_nd(VecZType &x, NumZ y, const Container<NumN> &d,
                                               const VecZType2 &a) {
        NumN n = d.size();
        MATH21_ASSERT(!x.isEmpty() && x.size() == n)
        Container<NumN> k(n);
        k(n) = 1;
        NumN i;
        for (i = n - 1; i >= 1; --i) {
            k(i) = d(i + 1) * k(i + 1);
        }
        y = y - (NumZ) a(n);
        for (i = 1; i <= n; ++i) {
            x(i) = y / k(i) + a(i);
            y = y % k(i);
        }
    }

    // d is shape, a is start index.
    // y-a(n) = sum ((x(i)-a(i))*k(i))
    template<template<typename> class Container,
            template<typename> class Container2, typename NumZType>
    void math21_operator_number_index_1d_to_tensor_nd(Container2<NumZType> &x, NumZ y, const Container<NumN> &d) {
        Container<NumZType> a(d.size());
        a = 1;
        math21_operator_number_index_1d_to_nd(x, y, d, a);
    }

    // see math21_device_index_replace_inc
    // replace A by R where A(i) = x.
    template<typename T, template<typename> class Container, typename VecType>
    void math21_operator_container_replace_inc(const Container<T> &A, Container<T> &B, const VecType &R, const T &x) {
        MATH21_ASSERT(B.size() == A.size())
        NumN j = 1;
        for (NumN i = 1; i <= B.size(); ++i) {
            if (A(i) == x) {
                B(i) = R(j);
                ++j;
            } else {
                B(i) = A(i);
            }
        }
        MATH21_ASSERT(j - 1 == R.size(), "j is " << j);
    }

    // replace A by R where A(i) = x.
    template<typename T, template<typename> class Container, typename VecType>
    void math21_operator_container_replace_by_same_pos(const Container<T> &A, Container<T> &B, const VecType &R,
                                                       const T &x) {
        MATH21_ASSERT(B.size() == A.size())
        MATH21_ASSERT(R.size() == A.size())
        for (NumN i = 1; i <= B.size(); ++i) {
            if (A(i) == x) {
                B(i) = R(i);
            } else {
                B(i) = A(i);
            }
        }
    }

    // normal mode
    // number to representation from right, x, i(k) in [1, ...].
    template<template<typename> class Container,
            template<typename> class Container2>
    void math21_operator_number_num_to_index_right(NumN x, Container<NumN> &i, const Container2<NumN> &d) {
        MATH21_ASSERT(d.size() == i.size())
        x = x - 1;
        NumN n = d.size();
        for (NumN k = n; k >= 1; --k) {
            i(k) = x % d(k) + 1;
            x = x / d(k);
        }
    }

    // number to representation from left, x, i(k) in [1, ...].
    template<template<typename> class Container,
            template<typename> class Container2>
    void math21_operator_number_num_to_index_left(NumN x, Container<NumN> &i, const Container2<NumN> &d) {
        printf("Check this please! left is changed to right in code.");
        MATH21_ASSERT(d.size() == i.size())
        x = x - 1;
        NumN n = d.size();
        for (NumN k = 1; k <= n; ++k) {
            i(k) = x % d(k) + 1;
            x = x / d(k);
        }
    }

    // representation to number from right, x, i(k) in [1, ...].
    template<template<typename> class Container,
            template<typename> class Container2>
    NumN math21_operator_number_index_to_num_right(const Container<NumN> &i, const Container2<NumN> &d) {
        MATH21_ASSERT(d.size() == i.size())
        NumN x;
        x = 0;
        NumN n = d.size();
        for (NumN k = 1; k <= n; ++k) {
            x = x * d(k) + i(k) - 1;
        }
        x = x + 1;
        return x;
    }

    // representation to number from left, x, i(k) in [1, ...].
    template<template<typename> class Container,
            template<typename> class Container2>
    NumN math21_operator_number_index_to_num_left(const Container<NumN> &i, const Container2<NumN> &d) {
        MATH21_ASSERT(d.size() == i.size())
        NumN x;
        x = 0;
        NumN n = d.size();
        for (NumN k = n; k >= 1; --k) {
            x = x * d(k) + i(k) - 1;
        }
        x = x + 1;
        return x;
    }

    template<typename VecType, template<typename> class Container>
    NumN math21_operator_container_2d_size(const Container<VecType> &v) {
        NumN k = 0;
        NumN n = v.size();
        for (NumN i = 1; i <= n; ++i) {
            k = k + v(i).size();
        }
        return k;
    }

    template<typename VecType, template<typename> class Container>
    NumN math21_operator_container_2d_element_size_max(const Container<VecType> &v) {
//        NumN k = 0;
        NumN size = 0;
        NumN n = v.size();
        for (NumN i = 1; i <= n; ++i) {
            if (v(i).size() > size) {
//                k = i;
                size = v(i).size();
            }
        }
        return size;
    }

    template<typename VecType, template<typename> class Container>
    void math21_operator_container_element_setSize(Container<VecType> &v, NumN k) {
        NumN n = v.size();
//#pragma omp parallel for
        for (NumN i = 1; i <= n; ++i) {
            v.at(i).setSize(k);
        }
    }

    template<typename T>
    T math21_operator_number_clip(const T &x0, const NumR &min, const NumR &max) {
        T x = x0;
        if (x > max) {
            x = max;
        } else if (x < min) {
            x = min;
        }
        return x;
    }

}