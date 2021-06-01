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
    void math21_opt_adam(
            TenR &x,
            void (*grad)(const TenR &x, TenR &dy, NumN iter, void *grad_data),
            void *grad_data,
            void (*callback)(const TenR &x_cur, const TenR &gradient, NumN iter, void *callback_data),
            void *callback_data,
            NumN num_iters = 100,
            NumR step_size = 0.001, NumR b1 = 0.9, NumR b2 = 0.999, NumR eps = 1e-8);
}