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

#include "inner_cc.h"
#include "op_mean.h"

namespace math21 {
    namespace ad {
        // see op_sum
        NumN op_mean::cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const {
            math21_tool_assert(X.size() == 1 || X.size() == 3);
            math21_tool_assert(X(1) == x);
            auto px = ad_point::to_point(x);
            auto pdy = ad_point::to_point(dy);
            if (X.size() == 1) {
                auto pk = 1 / ad_get_size(px);
                auto value = ad_kx(pk, pdy);
                auto pdx = ad_create_using_shape(value, ad_get_shape(px));
                return pdx.id;
            } else {
                NumB isKeepingDims;
                auto &_isKeepingDims = data(X(3)).getValue();
                MATH21_ASSERT(_isKeepingDims.isScalarInMath());
                isKeepingDims = (NumB) _isKeepingDims(1);
                if (!isKeepingDims) {
                    auto paxes = ad_point::to_point(X(2));
                    auto d = ad_get_shrink_shape_keeping_dim(px, paxes);
                    pdy = ad_share_reshape(pdy, d);
                }
                auto pk = ad_get_size(pdy) / ad_get_size(px);
                auto value = ad_kx(pk, pdy);
                auto pdx = ad_tensor_broadcast(value, ad_get_shape(px));
                return pdx.id;
            }
        }

        // X = {x, axes, isKeepingDims}
        void op_mean::fv(const Set &X, const Set &Y, VariableMap &data) const {
            MATH21_ASSERT(X.size() == 1 || X.size() == 3);
            auto &y = data.at(Y(1)).getValue();
            auto &x = data(X(1)).getValue();
            VecN axes;
            NumB isKeepingDims = 0;
            if (X.size() == 1) {
            } else {
                axes = data(X(2)).getValue();
                auto &_isKeepingDims = data(X(3)).getValue();
                MATH21_ASSERT(_isKeepingDims.isScalarInMath());
                isKeepingDims = (NumB) _isKeepingDims(1);
            }
            math21_op_mean(x, y, axes, isKeepingDims);
        }
    }
}