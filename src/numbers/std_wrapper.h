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

#include <string>
#include <vector>

#ifdef MATH21_NO_EXCEPTIONS
#else
#include <stdexcept>
#endif

#include "inner.h"
#include "number.h"
#include "_assert.h"


namespace math21 {

    /////////////////
    template<typename T>
    void math21_tool_std_vector_resize(std::vector<T> &A, NumN size) {
#ifdef MATH21_NO_EXCEPTIONS
        A.resize(size);
#else
        try {
            A.resize(size);
        }
        catch (const std::length_error &le) {
            MATH21_ASSERT(0)
        }
#endif
    }

    void math21_tool_std_string_resize(std::string &m, NumN size);
}