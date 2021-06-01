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
#include "../print/files.h"
#include "../memory/files.h"

namespace math21 {

///////////////////////////////// CUDA ////////////////////////////////////

#ifndef MATH21_FLAG_USE_CUDA

    static inline void math21_cuda_throw_no_cuda() {
                MATH21_ASSERT(0, "The library is compiled without CUDA support");
            }

#else // MATH21_FLAG_USE_CUDA

    static inline void math21_cuda_throw_no_cuda() {
        MATH21_ASSERT(0,
                      "The called functionality is disabled for current build or platform");
    }

#endif

}