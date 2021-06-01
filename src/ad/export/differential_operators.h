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
        void ad_grad_clear();

        ad_point ad_grad(ad_point x, ad_point y);

        ad_point grad(ad_point x, ad_point y);

        ad_point egrad(ad_point x, ad_point y);

        ad_point ad_jacobian_one_graph(ad_point x, ad_point y);

        ad_point ad_jacobian(ad_point x, ad_point y);

        ad_point ad_hessian(ad_point x, ad_point y);

        ad_point ad_hessian_vector_product(ad_point x, ad_point y, ad_point vector);

        void ad_fv(ad_point y);

        void ad_fv(ad_point x, ad_point y);
    }
}