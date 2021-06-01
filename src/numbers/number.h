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

#include <typeinfo>
#include <iomanip>
#include <cfloat>
#include <complex>
#include "number_c.h"

namespace math21 {

#define NumZ8_MIN (-0x7f-1) // -2^7
#define NumZ8_MAX 0x7f // 2^7-1
#define NumN8_MAX 0xff /* 255U, 2^8-1 */
#define NumN16_MAX 0xffff /* 65535U */
#define NumZ32_MIN (-0x7fffffff-1)
#define NumZ32_MAX 0x7fffffff
#define NumN32_MAX 0xffffffff  /* 4294967295U */
#define NumZ64_MIN  (-0x7fffffffffffffffLL-1) //-9223372036854775808
#define NumZ64_MAX 0x7fffffffffffffffLL //9223372036854775807LL
#define NumN64_MAX 0xffffffffffffffffULL /* 18446744073709551615ULL */

#define NumR_MAX DBL_MAX

    ///////////////

#define MATH21_MAX (NumZ32_MAX)
#define MATH21_MIN (NumZ32_MIN)

    ///////////////

    /* Some useful constants.  */
#define MATH21_E        2.7182818284590452354    /* e */
#define MATH21_LOG2E    1.4426950408889634074    /* log_2 e */
#define MATH21_LOG10E    0.43429448190325182765    /* log_10 e */
#define MATH21_LN2        0.69314718055994530942    /* log_e 2 */
#define MATH21_LN10        2.30258509299404568402    /* log_e 10 */
#define MATH21_PI        3.14159265358979323846    /* pi */
#define MATH21_PI_2        1.57079632679489661923    /* pi/2 */
#define MATH21_PI_4        0.78539816339744830962    /* pi/4 */
#define MATH21_1_PI        0.31830988618379067154    /* 1/pi */
#define MATH21_2_PI        0.63661977236758134308    /* 2/pi */
#define MATH21_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define MATH21_SQRT2    1.41421356237309504880    /* sqrt(2) */
#define MATH21_SQRT1_2    0.70710678118654752440    /* 1/sqrt(2) */

#define MATH21_MIN_POSITIVE_NUMR    1e-20

    ///////////////
//
    struct Interval;
    struct Interval2D;

    template<typename T>
    class Seqce;

    typedef Seqce<NumN> SeqceN;
    typedef Seqce<NumZ> SeqceZ;
    typedef Seqce<NumR> SeqceR;
//    typedef SeqceN Sequence;

    template<typename T>
    struct _Set;

    typedef _Set<NumN> SetN;
    typedef _Set<NumZ> SetZ;
    typedef _Set<NumR> SetR;
    typedef SetN Set;

    // should deprecate, and use seqce instead.
    template<typename T>
    struct _Sequence;
    typedef _Sequence<NumN> SequenceN;
    typedef _Sequence<NumZ> SequenceZ;
    typedef _Sequence<NumR> SequenceR;
    typedef SequenceN Sequence;

    template<typename T, typename S>
    struct _Map;
    typedef _Map<NumN, NumN> MapNN;
    typedef _Map<NumZ, NumZ> MapZZ;
    typedef _Map<NumR, NumR> MapRR;
    typedef MapNN Map;

    typedef std::complex<NumR> NumC;

    template<typename T>
    class Tensor;

    typedef Tensor<NumN> TenN;
    typedef Tensor<NumZ> TenZ;
    typedef Tensor<NumR> TenR;
    typedef Tensor<NumC> TenC;
    typedef Tensor<NumB> TenB;
    typedef Tensor<NumN8> TenN8;
    typedef Tensor<std::string> TenStr;
    typedef Tensor<NumR32> TenR32;
    typedef Tensor<NumSize> TenSize;

    typedef TenN MatN;
    typedef TenZ MatZ;
    typedef TenR MatR;
    typedef TenC MatC;
    typedef TenB MatB;
    typedef TenN8 MatN8;
    typedef TenStr MatStr;
    typedef TenR32 MatR32;
    typedef TenSize MatSize;

    typedef MatN VecN;
    typedef MatZ VecZ;
    typedef MatR VecR;
    typedef MatB VecB;
    typedef MatN8 VecN8;
    typedef MatStr VecStr;
    typedef MatR32 VecR32;
    typedef MatSize VecSize;

    // please use less frequently
    template<typename T>
    class IndexFunctional;

    typedef IndexFunctional<NumN> ShiftedTenN;
    typedef IndexFunctional<NumZ> ShiftedTenZ;
    typedef IndexFunctional<NumR> ShiftedTenR;

    typedef ShiftedTenN ShiftedMatN;
    typedef ShiftedTenZ ShiftedMatZ;
    typedef ShiftedTenR ShiftedMatR;

    typedef ShiftedMatN ShiftedVecN;
    typedef ShiftedMatZ ShiftedVecZ;
    typedef ShiftedMatR ShiftedVecR;

    namespace ad {
        struct ad_point;
    }

    class m21object;
}