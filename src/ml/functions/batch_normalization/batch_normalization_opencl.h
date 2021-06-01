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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_OPENCL

void math21_ml_batchnormalization_backward_mu_fast_opencl(PointerFloatInputWrapper dX_hat, PointerFloatInputWrapper variance, int mini_batch_size, int features_size, int in_class_size, PointerFloatWrapper dmu);

void
math21_ml_batchnormalization_backward_sigma_square_fast_opencl(PointerFloatInputWrapper X, PointerFloatInputWrapper dX_hat,
                                                           PointerFloatInputWrapper mu,
                                                           PointerFloatInputWrapper variance, int mini_batch_size,
                                                           int features_size, int in_class_size,
                                                           PointerFloatWrapper dvariance);
void math21_ml_batchnormalization_backward_input_opencl(PointerFloatInputWrapper X, PointerFloatInputWrapper mu, PointerFloatInputWrapper variance,
                                                        PointerFloatInputWrapper dmu, PointerFloatInputWrapper dvariance, int mini_batch_size,
                                                        int features_size, int in_class_size, PointerFloatWrapper dX_hat);

#endif

#ifdef __cplusplus
}
#endif
