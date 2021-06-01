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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

NumR math21_test_c_ad_sin_dn(NumR x, NumN order);

NumR math21_test_c_ad_sin_taylor_appr_dn(NumR x, NumN order);

m21point math21_test_ad_tanh_like_dn(m21point px, NumN order);

m21point math21_test_ad_logsumexp_like(m21point px);

// deprecate, use the following
m21point math21_test_ad_gmm_log_likelihood_dn(m21point params0, m21point data0,
                                              NumN n_component, NumN n_feature, NumN order);

m21point math21_test_ad_get_f_gmm_log_likelihood(m21point params0, m21point data0,
                                                 NumN n_component, NumN n_feature);

m21point math21_test_ad_get_f_gmm_log_likelihood_with_order(m21point params0, m21point data0,
                                                            NumN n_component, NumN n_feature, NumB isECorder);

m21point math21_test_ad_get_f_rnn_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size);

m21point math21_test_ad_get_f_lstm_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size);

m21point math21_test_ad_get_f_rnn_part_log_likelihood(
        m21point logprobs, m21point data_targets);

m21point math21_point_ad_grad(m21point x, m21point y);

m21point math21_point_ad_hessian_vector_product(m21point x, m21point y, m21point v);

void math21_point_ad_fv(m21point x);

void math21_ad_clear_graph();

#ifdef __cplusplus
}
#endif
