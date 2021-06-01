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

#include "matrix_cuda.h"

void math21_matrix_multiply_k1AB_add_k2C_similar_cuda(int TA, int TB, int nr_C, int nc_C, int n_common, float alpha,
                                                      const float *A_gpu, int lda,
                                                      const float *B_gpu, int ldb,
                                                      float beta,
                                                      float *C_gpu, int ldc) {
    cublasHandle_t handle = math21_cuda_blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N), nc_C, nr_C, n_common, &alpha, B_gpu, ldb, A_gpu,
                                     lda, &beta, C_gpu, ldc);
    math21_cuda_cublas_check_error(status);
}
