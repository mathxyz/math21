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

    namespace detail_li {

        // T is Num, pointer, or std::string type.
        // return index of s where found x, or 0 not found.
        // T and S may not be same.
        //// Note: s must be sorted first!
        template<template<typename> class Container, typename T, typename S, typename Compare>
        NumN binary_search(const Container<T> &s, const S &x, NumN n, const Compare &f) {
            MATH21_ASSERT(n <= s.size())
            if (n == 0) {
                return 0;
            }
            NumZ a, b;
            NumZ c;
            a = 1;
            b = n;
            while (a <= b) {
                c = (a + b) / 2;
                if (f.compare(s(c), x) == 0) {
                    return (NumN) c;
                }
                if (f.compare(s(c), x) > 0) {
                    b = c - 1;
                }
                if (f.compare(s(c), x) < 0) {
                    a = c + 1;
                }
            }
            return 0;
        }

        template<typename T>
        struct Compare_search_index {
            const Seqce <T> &s;

            explicit Compare_search_index(const Seqce <T> &s) : s(s) {
            }

            NumZ compare(NumN a, const T &b) const {
                if (s(a) == b) {
                    return 0;
                } else if (s(a) > b) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

        void li_test_binary_search();

        void li_test_binary_search_2();

        void li_test_binary_search_3();

    }
}