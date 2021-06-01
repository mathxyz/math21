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

#include "array.h"

namespace math21 {
    template<typename T>
    void Array<T>::swap(Array &B) {
        math21_tool_assert(isBasicType());
        MATH21_ASSERT(isIndependent(), "swapping data which is not independent! (Not test)");
        autoBuffer.swap(B.autoBuffer);
        m21_swap(n, B.n);
        m21_swap(v, B.v);
    }
}