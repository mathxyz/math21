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

#include "../../point/files.h"
#include "rnn.h"
#include "test_01_c.h"

using namespace math21;
using namespace ad;

m21point math21_test_ad_get_f_rnn_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != params.type
        || math21_type_get<ad_point>() != data_inputs.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto py = ad_rnn_predict(
            math21_cast_to_T<ad_point>(params),
            math21_cast_to_T<ad_point>(data_inputs),
            input_size,
            state_size,
            output_size);
    auto y = new ad_point();
    *y = py;
    return math21_cast_to_point(*y);
}

m21point math21_test_ad_get_f_lstm_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != params.type
        || math21_type_get<ad_point>() != data_inputs.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto py = ad_lstm_predict(
            math21_cast_to_T<ad_point>(params),
            math21_cast_to_T<ad_point>(data_inputs),
            input_size,
            state_size,
            output_size);
    auto y = new ad_point();
    *y = py;
    return math21_cast_to_point(*y);
}

m21point math21_test_ad_get_f_rnn_part_log_likelihood(
        m21point logprobs, m21point data_targets) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != logprobs.type
        || math21_type_get<TenR>() != data_targets.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto py = ad_rnn_part_log_likelihood(
            math21_cast_to_T<ad_point>(logprobs),
            math21_cast_to_T<TenR>(data_targets));
    auto y = new ad_point();
    *y = py;
    return math21_cast_to_point(*y);
}
