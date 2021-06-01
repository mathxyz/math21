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
    namespace ad {

        // natural logarithm function
        ad_point ad_log(const ad_point &x);

        ad_point ad_log(const ad_point &x, NumR base);

        ad_point ad_sin(const ad_point &p);

        ad_point ad_cos(const ad_point &p);

        ad_point ad_sum(const ad_point &p);

        ad_point ad_sum(const ad_point &p, const ad_point &axes, NumB isKeepingDims = 0);

        ad_point ad_tensor_broadcast(const ad_point &p, const ad_point &d);

        ad_point ad_add(const ad_point &p1, const ad_point &p2);

        ad_point ad_add(const ad_point &p1, const ad_point &p2, const ad_point &p3);

        ad_point ad_negate(const ad_point &p);

        ad_point ad_subtract(const ad_point &p1, const ad_point &p2);

        ad_point ad_mul(const ad_point &p1, const ad_point &p2);

        ad_point ad_mul(const ad_point &p1, const ad_point &p2, const ad_point &p3);

        ad_point ad_divide(const ad_point &p1, const ad_point &p2);

        ad_point ad_kx(const ad_point &k, const ad_point &x);

        ad_point operator+(const ad_point &p1, const ad_point &p2);

        ad_point operator-(const ad_point &p1);

        ad_point operator-(const ad_point &p1, const ad_point &p2);

        ad_point operator*(const ad_point &p1, const ad_point &p2);

        ad_point operator/(const ad_point &p1, const ad_point &p2);

        ad_point ad_power(const ad_point &x, NumR k, NumR p);

        // natural exponential function
        ad_point ad_exp(const ad_point &x);

        ad_point exp(const ad_point &x);

        ad_point ad_exp(const ad_point &x, NumR base);

        ad_point ad_mat_trans(const ad_point &x);

        ad_point ad_mat_mul(const ad_point &p1, const ad_point &p2);

        ad_point ad_inner_product(const ad_point &p1, const ad_point &p2);

        ad_point dot(const ad_point &p1, const ad_point &p2);

        ad_point ad_at(const ad_point &p, NumN index);

        ad_point at(const ad_point &p, NumN index);

        ad_point ad_push(const ad_point &p);

        ad_point ad_pull(const ad_point &p);

        ad_point ad_create_using_shape(const ad_point &value, const ad_point &d);

        ad_point ad_mean(const ad_point &p);

        ad_point ad_mean(const ad_point &p, const ad_point &axes, const ad_point &isKeepingDims);
    }
}