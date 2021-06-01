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

OptAlphaPolicy math21_opt_learning_rate_policy_get_from_name(const char *s);

float math21_opt_get_alpha_by_policy(m21OptAlphaPolicyConfig *config);

void math21_optimization_adam_update_wrapper(PointerFloatWrapper x, PointerFloatWrapper neg_dx, PointerFloatWrapper m,
                                             PointerFloatWrapper v, float beta1, float beta2,
                                             float eps, float decay, float alpha, int x_size, int mini_batch_size,
                                             int t);

void math21_generic_optimization_adam_update_wrapper(
        PointerVoidWrapper x, PointerVoidWrapper neg_dx, PointerVoidWrapper m,
        PointerVoidWrapper v, NumR beta1, NumR beta2,
        NumR eps, NumR decay, NumR alpha, NumN x_size, NumN mini_batch_size,
        NumN t, NumN type);

#ifdef __cplusplus
}
#endif
