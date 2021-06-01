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

#include <cmath>
#include "inner.h"

namespace math21 {
    /*elementary math functions*/

    template<typename T>
    int xjisnan(T x) {
        if (std::isnan(x)) {
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T>
    NumB xjisfinite(T x) {
        if (std::isfinite((NumR) x)) {
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T>
    NumB xjisinf(T x) {
        if (std::isinf(x)) {
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T>
    M21_EXPORT T xjabs(const T &x) {
        if (x >= 0) {
            return x;
        } else {
            return -x;
        }
    }

    template<typename T>
    M21_EXPORT inline NumZ xjsameSign(const T &x, const T &y) {
        return y >= 0 ? (x >= 0 ? 1 : -1) : (x >= 0 ? -1 : 1);
    }

    template<typename T>
    M21_EXPORT NumZ xjsign(const T &x) {
        if (x >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

    template<typename T>
    M21_EXPORT NumZ xj_sign0(const T &x) {
        if (x > 0) {
            return 1;
        } else if (x == 0) {
            return 0;
        } else {
            return -1;
        }
    }

    // f(x, y) = sameSign(x,y)*x
    template<class T>
    M21_EXPORT inline T xjchangeSign(const T &x, const T &y) {
        return xjsameSign(x, y) * x;
    }

    template<typename T>
    M21_EXPORT T xjsquare(const T &x) {
        return x * x;
    }

    template<typename T>
    M21_EXPORT NumR xjatan(const T &x) {
        return std::atan(x);
    }

    // return arctan(x/y)
    // Principal arc tangent of x/y, in the interval [-pi,+pi] radians.
    template<typename T1, typename T2>
    M21_EXPORT NumR xjatan2(const T1 &x, const T2 &y) {
        return std::atan2(x, y);
    }

    template<typename T>
    M21_EXPORT NumR xjtan(T x) {
        return std::tan(x);
    }

    template<typename T>
    M21_EXPORT NumR xjexp(T x) {
        return std::exp(x);
    }

    M21_EXPORT NumZ xjfloor(NumR x);

    M21_EXPORT NumZ xjceil(NumR x);

    // permutation with repetition: f(r) = n^r
    template<typename T, typename S>
    M21_EXPORT NumR xjpow(const T &n, const S &r) { return pow(n, r); }

    // permutation without repetition, f(r) = n!/(n-r)!
    M21_EXPORT NumN xjfactorial_similar(NumN n, NumN r);

    M21_EXPORT NumN xjfactorial(NumN n);

    // combination without repetition, f(r) = n!/(r!(n-r)!)
    // property: f(r) = f(n-r)
    M21_EXPORT NumN xj_n_choose_r(NumN n, NumN r);

    // Let us say there are five flavors of icecream: banana, chocolate, lemon, strawberry and vanilla.
    // We can have three scoops. How many variations will there be?
    // combination with repetition
    // f(r) = xj_n_choose_r(r+n-1, r)
    M21_EXPORT NumN xj_combination_rep(NumN n, NumN r);

    M21_EXPORT NumR xjpoint2angle(NumR x, NumR y);

    template<typename T>
    M21_EXPORT T xjadd(const T &x, const T &y) {
        return x + y;
    }

    template<typename T>
    M21_EXPORT T xjsubtract(const T &x, const T &y) {
        return x - y;
    }

    template<typename T>
    M21_EXPORT T xjmultiply(const T &x, const T &y) {
        return x * y;
    }

    template<typename T>
    M21_EXPORT T xjdivide(const T &x, const T &y) {
        return x / y;
    }

    template<typename T>
    M21_EXPORT T xjmodulus(const T &x, const T &y) {
        return x % y;
    }

    template<typename T>
    M21_EXPORT T xjnegate(const T &x) {
        return -x;
    }

    template<typename T>
    M21_EXPORT NumR xjlog(T x) {
        return std::log(x);
    }

    template<typename T>
    M21_EXPORT NumR xjsin(T x) {
        return std::sin(x);
    }

    template<typename T>
    M21_EXPORT NumR xjcos(T x) {
        return std::cos(x);
    }

    template<typename T>
    M21_EXPORT NumR xjsqrt(T x) {
        return std::sqrt(x);
    }

    template<typename T>
    M21_EXPORT NumR xjacos(T x) {
        return std::acos(x);
    }

    template<typename T>
    M21_EXPORT NumR xjasin(T x) {
        return std::asin(x);
    }

    template<typename T, typename S>
    M21_EXPORT T xjmin(const T &x, const S &y) {
        return x < y ? x : y;
    }

    template<typename T>
    M21_EXPORT T xjmin(const T &x1, const T &x2, const T &x3) {
        return xjmin(xjmin(x1, x2), x3);
    }

    template<typename T, typename S>
    M21_EXPORT T xjmax(const T &x, const S &y) {
        return x > y ? x : y;
    }

    template<typename T>
    M21_EXPORT T xjmax(const T &x1, const T &x2, const T &x3) {
        return xjmax(xjmax(x1, x2), x3);
    }

    template<typename T, typename S>
    M21_EXPORT NumR xjisLarger(const T &x, const S &y) {
        return x > y ? 1 : 0;
    }

    template<typename T, typename S>
    M21_EXPORT NumB xjisNotLarger(const T &x, const S &y) {
        return x <= y ? 1 : 0;
    }

    template<typename T, typename S>
    M21_EXPORT NumB xjisSmaller(const T &x, const S &y) {
        return x < y ? 1 : 0;
    }

    template<typename T, typename S>
    M21_EXPORT NumB xjisNotSmaller(const T &x, const S &y) {
        return x >= y ? 1 : 0;
    }

    // see matlab mod.
    // y = mod(x, m), y = x % m
    template<typename T>
    M21_EXPORT NumR xjmod(const T &x, const NumR &m) {
        if (m == 0) {
            return x;
        }
        NumR y = x - m * floor(x / m);
        return y;
    }

    template<typename T>
    M21_EXPORT NumR xjradian2degree(const T &x) {
        return (NumR) (x * 180 / XJ_PI);
    }

    template<typename T>
    M21_EXPORT NumR xjdegree2radian(const T &x) {
        return (NumR) (x * XJ_PI / 180);
    }

    template<typename T>
    M21_EXPORT int xjmathInsideClose(T &x, T &a, T &b) {
        if (x < a || x > b)return 0;
        else return 1;
    }

    template<typename T>
    M21_EXPORT int xjmathInsideOpen(T &x, T &a, T &b) {
        if (x <= a || x >= b)return 0;
        else return 1;
    }

    template<typename T>
    M21_EXPORT int xjmathIs0to9(T &c) {
        if ((char) c < '0' || (char) c > '9')return 0;
        else return 1;
    }

// user should check c.
    template<typename T>
    M21_EXPORT int xjmathTo0to9(T &c) {
        return (char) c - '0';
    }

    // is x in [a, b]
    template<typename T1, typename T2, typename T3>
    M21_EXPORT NumB xjIsIn(const T1 &x, const T2 &a, const T3 &b) {
#ifdef MATH21_WIN_MSVC
        if ((NumR) x < (NumR) a || (NumR) x > (NumR) b) {
            return 0;
        }
#else
        if (x < a || x > b) {
            return 0;
        }
#endif
        return 1;
    }

    // make sure x is in [a, b]
    template<typename T>
    M21_EXPORT T xjconstrain(const T &x, const T &a, const T &b) {
        if (x < a)return a;
        if(x>b)return b;
        return x;
    }

    template<typename T, typename S>
    M21_EXPORT S xjmathScale(T x, T a, T b, S a2, S b2) {
        MATH21_ASSERT(b - a != 0, "b-a==0");
        if (x < a) {
            return a2;
        }
        if (x > b) {
            return b2;
        }
        return (S) (a2 + (x - a) * (b2 - a2) / (b - a));
    }

    M21_EXPORT NumB xjIsEven(const NumZ &x);

    M21_EXPORT NumZ xjR2Z_upper(const NumR &x);

    M21_EXPORT NumZ xjR2Z_lower(const NumR &x);

    // error ratio >= 0, b-|b|*ratio <= a <= b + |b|*ratio
    M21_EXPORT NumB xjIsApproximatelyEqual(const NumR &a, const NumR &b, NumR ratio);

    // round to n decimal places
    template<typename T>
    M21_EXPORT NumR xjround(const T &x, NumN n) {
        return round(x * pow(10, n)) / pow(10, n);
    }

    // see math21_operator_number_index_1d_to_nd
    // d is shape, index from 0
    M21_EXPORT inline void xj_index_1d_to_nd(NumN *x, NumN y, const NumN *d, NumN n) {
        x -= 1;
        d -= 1;
        NumN i;
        for (i = n; i >= 1; --i) {
            x[i] = y % d[i];
            y = y / d[i];
        }
    }

    // see math21_device_index_nd_to_1d
    // d is shape, index from 0
    M21_EXPORT inline void xj_index_nd_to_1d(const NumN *x, NumN *py, const NumN *d, NumN n) {
        x -= 1;
        d -= 1;
        NumN i;
        NumN y = 0;
        for (i = 1; i <= n; ++i) {
            y *= d[i];
            y += x[i];
        }
        *py = y;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_1d_to_5d(NumN *x1, NumN *x2, NumN *x3, NumN *x4, NumN *x5, NumN y,
                                             NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        *x5 = y % d5;
        y = y / d5;
        *x4 = y % d4;
        y = y / d4;
        *x3 = y % d3;
        y = y / d3;
        *x2 = y % d2;
        y = y / d2;
        *x1 = y % d1;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_5d_to_1d(NumN x1, NumN x2, NumN x3, NumN x4, NumN x5, NumN *py,
                                             NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        NumN y = 0;
        y *= d1;
        y += x1;
        y *= d2;
        y += x2;
        y *= d3;
        y += x3;
        y *= d4;
        y += x4;
        y *= d5;
        y += x5;
        *py = y;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_1d_to_3d(NumN *x1, NumN *x2, NumN *x3, NumN y,
                                             NumN d1, NumN d2, NumN d3) {
        *x3 = y % d3;
        y = y / d3;
        *x2 = y % d2;
        y = y / d2;
        *x1 = y % d1;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_3d_to_1d(NumN x1, NumN x2, NumN x3, NumN *py,
                                             NumN d1, NumN d2, NumN d3) {
        NumN y = 0;
        y *= d1;
        y += x1;
        y *= d2;
        y += x2;
        y *= d3;
        y += x3;
        *py = y;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_1d_to_2d(NumN *x1, NumN *x2, NumN y,
                                             NumN d1, NumN d2) {
        *x2 = y % d2;
        y = y / d2;
        *x1 = y % d1;
    }

    // d is shape, index from 0
    M21_EXPORT inline void xj_index_2d_to_1d(NumN x1, NumN x2, NumN *py,
                                             NumN d1, NumN d2) {
        NumN y = 0;
        y *= d1;
        y += x1;
        y *= d2;
        y += x2;
        *py = y;
    }

    template<typename T, typename S>
    T math21_number_not_less_than(const T &x, const S &y) {
        if (x < y) {
            return (T) y;
        }
        return x;
    }

    void math21_number_get_from_and_to(NumN n, NumZ &from, NumZ &to);

    NumB math21_number_check_from_and_to(NumN n, NumZ from, NumZ to);

    void math21_number_get_from_only_with_check(NumN n0, NumZ &from);

    void math21_number_get_from_and_num_with_check(NumN n0, NumZ &from, NumN &num);
}