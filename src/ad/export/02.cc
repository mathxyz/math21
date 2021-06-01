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

#include "../functions_01/files.h"
#include "../functions_02/files.h"
#include "../functions_03/files.h"
#include "../differential.h"
#include "point.h"
#include "01.h"
#include "02.h"

namespace math21 {
    namespace ad {
        // f = tanh(x) = sinh(x)/cosh(x) = (exp(2x)-1)/(exp(2x)+1)
        // y = exp(-2.0 * x), f = (1.0 - y) / (1.0 + y)
        ad_point ad_tanh(const ad_point &x) {
            auto y = ad_exp(-2.0 * x);
            return (1.0 - y) / (1.0 + y);
        }

        ad_point tanh(const ad_point &x) {
            return ad_tanh(x);
        }

        // y = exp(-2.0 * x), f = 1.0 / (1.0 + y)
        // <=> y = f(x) = 0.5 * (ad_tanh(x) + 1.0)
        ad_point ad_sigmoid_from_tanh(const ad_point &x) {
            auto y = ad_exp(-2.0 * x);
            return 1.0 / (1.0 + y);
        }

        // y = exp(-x), f = 1.0 / (1.0 + y)
        ad_point ad_sigmoid_from_logistic(const ad_point &x) {
            auto y = ad_exp(-x);
            return 1.0 / (1.0 + y);
        }

        // axes: same as numpy, but from index 1
        ad_point ad_logsumexp(const ad_point &x, const VecN &axes, NumB isKeepingDims) {
            op_logsumexp logsumexp(axes, isKeepingDims);
            Function &function = logsumexp;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mvn_logpdf(const ad_point &x, const ad_point &mean, const ad_point &covariance) {
            op_mvn_logpdf logpdf;
            Function &function = logpdf;
            NumN y = function.evaluate(x.id, mean.id, covariance.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_vec_share_sub_from_to(const ad_point &x, NumZ from, NumZ to) {
            op_vec_share_sub _op_vec_share_sub(from, to);
            Function &function = _op_vec_share_sub;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_vec_share_sub_offset(const ad_point &x, NumN offset, NumN n) {
            return ad_vec_share_sub_from_to(x, offset + 1, offset + n);
        }

        ad_point ad_get_shape(const ad_point &x) {
            op_get_shape _op_get_shape;
            Function &function = _op_get_shape;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_get_size(const ad_point &x) {
            op_get_size _;
            Function &f = _;
            NumN y = f.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_get_shrink_shape_keeping_dim(const ad_point &x, const ad_point &axes) {
            op_get_shrink_shape_keeping_dim _;
            Function &f = _;
            NumN y = f.evaluate(x.id, axes.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_share_reshape(const ad_point &x, const ad_point &d) {
            op_share_reshape _op_share_reshape;
            Function &function = _op_share_reshape;
            NumN y = function.evaluate(x.id, d.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        // use Set to pack different vectors.
        ad_point ad_row_pack(const ad_point &p1, const ad_point &p2, const ad_point &p3) {
            _Set<ad_point> x;
            x.add(p1);
            x.add(p2);
            x.add(p3);
            return ad_row_pack(x);
        }

        // row pack tensors
        ad_point ad_row_pack(const _Set<ad_point> &x) {
            op_row_pack _op_row_pack;
            Function &function = _op_row_pack;
            Set s;
            for (NumN i = 1; i <= x.size(); ++i) {
                s.add(x(i).id);
            }
            NumN y = function.evaluate(s, ad_global_get_data());
            return ad_point(y, 0);
        }

        // row pack tensors
        ad_point ad_row_unpack_i(const ad_point &x, NumN pos) {
            op_row_unpack_i _op_row_unpack_i(pos);
            Function &function = _op_row_unpack_i;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        // row unpack tensor
        void ad_row_unpack(const ad_point &x, _Set<ad_point> &y) {
            y.clear();
            NumN n = ad_get_dim_i(x, 1);
            for (NumN i = 1; i <= n; ++i) {
                auto _ = ad_row_unpack_i(x, i);
                y.add(_);
            }
        }

        ad_point ad_log_normalize_in_ad_gmm_log_likelihood(const ad_point &x) {
            return x - ad_logsumexp(x);
        }

        // params: {proportions, means, covs_sqrt} with shape [n_component, n_component*n_feature, n_component*n_feature*n_feature]
        // params: {proportions, covs_sqrt, means} with shape [n_component, n_component*n_feature*n_feature, n_component*n_feature]
        // data: n_data * n_feature, with n_feature = 2
        ad_point ad_gmm_log_likelihood(const ad_point &params, const ad_point &data,
                                       NumN n_component, NumN n_feature, NumB isECorder) {
            MATH21_ASSERT(ad_get_value(params).size() ==
                          n_component + n_component * n_feature + n_component * n_feature * n_feature,
                          ""
                                  << "ad_get_value(params).size() = " << ad_get_value(params).size()
                                  << "\nn_component = " << n_component
                                  << "\nn_feature = " << n_feature
            )
            auto proportions = ad_vec_share_sub_from_to(params, 1, n_component);
            auto log_proportions = ad_log_normalize_in_ad_gmm_log_likelihood(proportions);
            ad_point means, covs_sqrt;
            if (isECorder) {
                means = ad_vec_share_sub_from_to(params, n_component + 1, n_component + n_component * n_feature);
                covs_sqrt = ad_vec_share_sub_from_to(params, n_component + n_component * n_feature + 1, -1);
            } else { // CE order
                covs_sqrt = ad_vec_share_sub_from_to(params, n_component + 1,
                                                     n_component + n_component * n_feature * n_feature);
                means = ad_vec_share_sub_from_to(params, n_component + n_component * n_feature * n_feature + 1, -1);
            }
            VecN d_mean(1);
            d_mean = n_feature;
            auto p_d_mean = ad_point(d_mean);
            VecN d_cov_sqrt(2);
            d_cov_sqrt = n_feature, n_feature;
            auto p_d_cov_sqrt = ad_point(d_cov_sqrt);
            _Set<ad_point> cluster_lls;

            for (NumN i = 1; i <= n_component; ++i) {
                auto log_proportion = ad_vec_share_sub_from_to(log_proportions, i, i);
                auto mean = ad_vec_share_sub_from_to(means, (i - 1) * n_feature + 1, i * n_feature);
                // maybe use std::shared_ptr to reduce memory
                mean = ad_share_reshape(mean, p_d_mean);
                auto cov_sqrt = ad_vec_share_sub_from_to(covs_sqrt,
                                                         (i - 1) * n_feature * n_feature + 1,
                                                         i * n_feature * n_feature);
//                ad_get_value(cov_sqrt).reshape(d_cov_sqrt);
                cov_sqrt = ad_share_reshape(cov_sqrt, p_d_cov_sqrt);
                auto cov_sqrt_t = ad_mat_trans(cov_sqrt);
                auto cov = ad_mat_mul(cov_sqrt_t, cov_sqrt);
                auto _ = log_proportion + ad_mvn_logpdf(data, mean, cov);
                cluster_lls.add(_);
            }
            VecN axes(1);
            axes = 1;
            return ad_sum(ad_logsumexp(ad_row_pack(cluster_lls), axes));
        }

        // repeats: The number of repetitions for each element.  `repeats` is broadcasted
        //        to fit the shape of the given axis.
        // axis: The axis along which to repeat values.  By default, use the
        //        flattened input array, and return a flat output array.
        ad_point ad_repeat(const ad_point &x, const ad_point &repeats, ad_point axis) {
            op_repeat _op_repeat;
            Function &function = _op_repeat;
            NumN y = function.evaluate(x.id, repeats.id, axis.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_undo_repeat_sum(const ad_point &x, const ad_point &repeats, ad_point axis) {
            op_undo_repeat_sum _op_undo_repeat_sum;
            Function &function = _op_undo_repeat_sum;
            NumN y = function.evaluate(x.id, repeats.id, axis.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_concatenate(const ad_point &x1, const ad_point &x2, ad_point axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            NumN y = function.evaluate(x1.id, x2.id, axis.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_concatenate(const ad_point &x1, const ad_point &x2, const ad_point &x3, ad_point axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            Set X;
            X.add(x1.id);
            X.add(x2.id);
            X.add(x3.id);
            X.add(axis.id);
            NumN y = function.evaluate(X, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_concatenate(const Seqce<ad_point> &xs, ad_point axis) {
            op_concatenate _op_concatenate;
            Function &function = _op_concatenate;
            Set X;
            for (NumN i = 1; i <= xs.size(); ++i) {
                X.add(xs(i).id);
            }
            X.add(axis.id);
            NumN y = function.evaluate(X, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_axis_i_sub_get(
                const ad_point &x, const ad_point &offset, const ad_point &di, const ad_point &axis) {
            op_axis_i_sub_get _;
            Function &function = _;
            Set X;
            X.add(x.id);
            X.add(offset.id);
            X.add(di.id);
            X.add(axis.id);
            NumN y = function.evaluate(X, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_axis_i_sub_set(
                const ad_point &x, const ad_point &offset, const ad_point &d_y, const ad_point &axis) {
            op_axis_i_sub_set _;
            Function &function = _;
            Set X;
            X.add(x.id);
            X.add(offset.id);
            X.add(d_y.id);
            X.add(axis.id);
            NumN y = function.evaluate(X, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_axis_swap(const ad_point &x, const ad_point &pos, const ad_point &pos2) {
            op_axis_swap _;
            Function &function = _;
            NumN y = function.evaluate(x.id, pos.id, pos2.id, ad_global_get_data());
            return ad_point(y, 0);
        }
    }
}