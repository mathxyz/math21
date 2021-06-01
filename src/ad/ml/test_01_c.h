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

m21point math21_test_ad_get_f_rnn_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size);

m21point math21_test_ad_get_f_lstm_predict(
        m21point params, m21point data_inputs,
        NumN input_size, NumN state_size, NumN output_size);

m21point math21_test_ad_get_f_rnn_part_log_likelihood(
        m21point logprobs, m21point data_targets);

#ifdef __cplusplus
}
#endif
