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
        ad_point ad_tanh(const ad_point &x);

        ad_point tanh(const ad_point &x);

        ad_point ad_sigmoid_from_tanh(const ad_point &x);

        ad_point ad_sigmoid_from_logistic(const ad_point &x);

        ad_point ad_logsumexp(const ad_point &x, const VecN &axes = VecN(), NumB isKeepingDims = 0);

        ad_point ad_mvn_logpdf(const ad_point &x, const ad_point &mean, const ad_point &covariance);

        ad_point ad_vec_share_sub_from_to(const ad_point &x, NumZ from, NumZ to);

        ad_point ad_vec_share_sub_offset(const ad_point &x, NumN offset, NumN n);

        ad_point ad_get_shape(const ad_point &x);

        ad_point ad_get_size(const ad_point &x);

        ad_point ad_get_shrink_shape_keeping_dim(const ad_point &x, const ad_point &axes);

        ad_point ad_share_reshape(const ad_point &x, const ad_point &d);

        ad_point ad_row_pack(const ad_point &p1, const ad_point &p2, const ad_point &p3);

        ad_point ad_row_pack(const _Set <ad_point> &x);

        void ad_row_unpack(const ad_point &x, _Set <ad_point> &y);

        ad_point ad_gmm_log_likelihood(const ad_point &params, const ad_point &data,
                                       NumN n_component, NumN n_feature, NumB isECorder = 1);

        ad_point ad_repeat(const ad_point &x, const ad_point &repeats, ad_point axis);

        ad_point ad_undo_repeat_sum(const ad_point &x, const ad_point &repeats, ad_point axis);


        ad_point ad_concatenate(const ad_point &x1, const ad_point &x2, ad_point axis);

        ad_point ad_concatenate(const ad_point &x1, const ad_point &x2, const ad_point &x3, ad_point axis);

        ad_point ad_concatenate(const Seqce <ad_point> &xs, ad_point axis);

        // x, y can be empty tensor
        // can get sub-tensor y from tensor x
        ad_point ad_axis_i_sub_get(const ad_point &x, const ad_point &offset, const ad_point &di, const ad_point &axis);

        ad_point ad_axis_i_sub_set(const ad_point &x, const ad_point &offset, const ad_point &d_y, const ad_point &axis);

        ad_point ad_axis_swap(const ad_point &x, const ad_point &pos, const ad_point &pos2);
    }
}