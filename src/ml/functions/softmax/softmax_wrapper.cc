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

#include "softmax_cpu.h"
#include "softmax_cuda.h"
#include "softmax_opencl.h"
#include "softmax_wrapper.h"

void math21_ml_function_softmax_tree_wrapper(
        PointerFloatWrapper input, int in_class_size, int mini_batch_size,
        int stride, float temp, PointerFloatWrapper output, m21tree hier) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_softmax_tree_cpu(
            input, in_class_size, mini_batch_size,
            stride, temp, output, hier);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_softmax_tree_cuda(
            input, in_class_size, mini_batch_size,
            stride, temp, output, hier);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_softmax_tree_opencl(
            input, in_class_size, mini_batch_size,
            stride, temp, output, hier);
#endif
}

void math21_ml_function_softmax_wrapper(PointerFloatWrapper input, int n, int mini_batch_size, int batch_offset, int groups,
                                        int group_offset, int stride, float temp, PointerFloatWrapper output) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_softmax_cpu(input, n, mini_batch_size, batch_offset, groups,
                                    group_offset, stride, temp, output);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_softmax_cuda(input, n, mini_batch_size, batch_offset, groups,
                                    group_offset, stride, temp, output);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_softmax_opencl(input, n, mini_batch_size, batch_offset, groups,
                                    group_offset, stride, temp, output);
#endif
}

void math21_ml_function_softmax_x_ent_wrapper(int n, PointerFloatWrapper pred, PointerFloatWrapper truth, PointerFloatWrapper delta, PointerFloatWrapper error) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_ml_function_softmax_x_ent_cpu(n, pred, truth, delta, error);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_ml_function_softmax_x_ent_cuda(n, pred, truth, delta, error);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_ml_function_softmax_x_ent_opencl(n, pred, truth, delta, error);
#endif
}