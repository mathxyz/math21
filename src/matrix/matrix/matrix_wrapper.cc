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

#include "matrix_cpu.h"
#include "matrix_cuda.h"
#include "matrix_opencl.h"
#include "matrix_wrapper.h"

void math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(int TA, int TB, int nr_C, int nc_C, int n_common, float k1,
                                                         PointerFloatInputWrapper A, int lda,
                                                         PointerFloatInputWrapper B, int ldb,
                                                         float k2,
                                                         PointerFloatWrapper C, int ldc) {
#if defined(MATH21_FLAG_USE_CPU)
    math21_matrix_multiply_k1AB_add_k2C_similar_cpu(TA, TB, nr_C, nc_C, n_common, k1, A, lda, B, ldb, k2, C, ldc);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_matrix_multiply_k1AB_add_k2C_similar_cuda(TA, TB, nr_C, nc_C, n_common, k1, A, lda, B, ldb, k2, C, ldc);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_matrix_multiply_k1AB_add_k2C_similar_opencl(TA, TB, nr_C, nc_C, n_common, k1, A, lda, B, ldb, k2, C, ldc);
#endif
}
