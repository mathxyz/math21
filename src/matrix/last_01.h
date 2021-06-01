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

#include <algorithm>
#include "inner.h"

namespace math21 {

    // deprecate
    namespace vec_detail {
        template<typename T>
        struct Comp {
            bool operator()(T i, T j) {
                return (i < j);
            }
        };
    }

    template<typename T>
    void Array<T>::sort() {
        math21_tool_assert(isBasicType());
        math21_algorithm_sort(*this);
    }

    template<typename T>
    template<typename Compare>
    void Array<T>::sort(const Compare &f) {
        math21_tool_assert(isBasicType());
        math21_algorithm_sort(*this, f);
    }

    template<typename T>
    void Seqce<T>::sort() {
        math21_algorithm_sort(*this);
    }

}