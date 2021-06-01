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
#include "net.h"

#ifdef __cplusplus
extern "C" {
#endif

float math21_ml_function_multinets_train(mlfunction_net **fnets, int n_net, m21data d, int interval);

void math21_ml_function_multinet_sync(mlfunction_net **fnets, int num_net, int interval);

#ifdef __cplusplus
}
#endif
