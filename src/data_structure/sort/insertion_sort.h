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

        template<template<typename> class Container, typename T, typename Compare>
        void insertion_sort_insert(Container<T> &s, NumZ i,
                                   Compare &f) {
            T c;
            c = s(i);
            NumZ j = i - 1;
            while (j >= 1 && f.compare(c, s(j)) < 0) {
                s(j + 1) = s(j);
                j--;
            }
            s(j + 1) = c;
        }

        // T is Num, pointer, or std::string type.
        template<template<typename> class Container, typename T, typename Compare>
        void insertion_sort(Container<T> &s, NumN n, Compare &f) {
            MATH21_ASSERT(n <= s.size())
            if (n <= 1) {
                return;
            }
            NumN i;
            for (i = 2; i <= n; ++i) {
                insertion_sort_insert(s, i, f);
            }
        }

        void li_test_insertion_sort();

        void li_test_insertion_sort_2();

        void li_test_insertion_sort_3();

    }
}