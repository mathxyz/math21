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

#include "number.h"
#include <cmath>
#include <sstream>

namespace math21 {

    template<typename T>
    T m21_log(const T &a) { return log(a); }

    template<typename T>
    T m21_sqr(const T &a) { return a * a; }

    template<typename T>
    T m21_abs(const T &x) {
        if (x >= 0) {
            return x;
        } else {
            return -x;
        }
    }

    template<typename T, typename S>
    NumR m21_pow(const T &a, const S &b) { return pow(a, b); }

    template<class T>
    const T &m21_max(const T &a, const T &b) { return b > a ? (b) : (a); }

    inline float m21_max(const double &a, const float &b) { return b > a ? (b) : float(a); }

    inline float m21_max(const float &a, const double &b) { return b > a ? float(b) : (a); }

    template<class T>
    const T &m21_min(const T &a, const T &b) { return b < a ? (b) : (a); }

    inline float m21_min(const double &a, const float &b) { return b < a ? (b) : float(a); }

    inline float m21_min(const float &a, const double &b) { return b < a ? float(b) : (a); }

    template<class T>
    T m21_sign(const T &a, const T &b) { return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); }

    inline float m21_sign(const float &a, const double &b) { return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); }

    inline float m21_sign(const double &a, const float &b) {
        return (float) (b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a));
    }

    template<class T>
    void m21_swap(T &a, T &b) {
        T dum = a;
        a = b;
        b = dum;
    }

    template<typename T>
    std::string math21_string_to_string(const T &x) {
        std::ostringstream oss;
        oss << x;
        return oss.str();
    }

    template<typename T, typename T2>
    std::string math21_string_to_string(const T &x, const T2 &x2) {
        std::ostringstream oss;
        oss << x << x2;
        return oss.str();
    }

    template<typename T, typename T2, typename T3>
    std::string math21_string_to_string(const T &x, const T2 &x2, const T3 &x3) {
        std::ostringstream oss;
        oss << x << x2 << x3;
        return oss.str();
    }

    template<typename T>
    void math21_string_to_type_generic(const std::string &s, T &num) {
        std::istringstream iss(s);
        iss >> num;
    }

    template<>
    void math21_string_to_type_generic(const std::string &s, std::string &num);

    NumR math21_string_to_NumR(const std::string &s);

    NumN math21_string_to_NumN(const std::string &s);

    NumZ math21_string_to_NumZ(const std::string &s);

    std::string math21_string_NumZ_to_string(NumZ x, NumN width = 0);

    std::string math21_string_replicate_n(NumN n, const std::string &s);

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2);

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3);

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                                          const std::string &s4);

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                                          const std::string &s4, const std::string &s5);

    NumB math21_string_is_equal(const char *str1, const char *str2);

    NumB math21_string_is_equal(const std::string &str1, const std::string &str2);

    template<typename LogType>
    void math21_string_log_2_string(const LogType &A, std::string &s) {
        std::ostringstream oss;
        A.log(oss);
        s = oss.str();
    }


    NumB math21_point_isEqual(const NumR &x, const NumR &y, NumR epsilon = 0);

    NumB math21_point_isEqual(const NumN &x, const NumN &y, NumR epsilon = 0);

    NumB math21_point_isEqual(const NumZ &x, const NumZ &y, NumR epsilon = 0);

    NumB math21_type_size_t_is_4_bytes();

    NumB math21_type_NumSize_is_4_bytes();
}