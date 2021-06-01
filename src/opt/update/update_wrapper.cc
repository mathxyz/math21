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

#include "update_cpu.h"
#include "update_cuda.h"
#include "update_opencl.h"
#include "update_wrapper.h"
#include "../../generic/files_c.h"

// deprecate, use math21_generic_optimization_adam_update_part_2_wrapper
void math21_optimization_adam_update_part_2_wrapper(int x_size, PointerFloatWrapper x, PointerFloatWrapper m, PointerFloatWrapper v, float beta1, float beta2,
                                                    float alpha, float eps, int t) {
    math21_generic_optimization_adam_update_part_2_wrapper(x_size, x, m, v, beta1, beta2, alpha, eps, t, m21_type_NumR32);
//#if defined(MATH21_FLAG_USE_CPU)
//    math21_optimization_adam_update_part_2_cpu(x_size, x, m, v, beta1, beta2, alpha, eps, t);
//#elif defined(MATH21_FLAG_USE_CUDA)
//    math21_optimization_adam_update_part_2_cuda(x_size, x, m, v, beta1, beta2, alpha, eps, t);
//#elif defined(MATH21_FLAG_USE_OPENCL)
//    math21_optimization_adam_update_part_2_opencl(x_size, x, m, v, beta1, beta2, alpha, eps, t);
//#endif
}