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

#include "../vector/files_c.h"
#include "matrix_cpu.h"

// C = k1*A*B + C
void math21_matrix_multiply_k1AB_add_k2C_similar_nn_cpu(int nr_C, int nc_C, int n_common, float k1,
                                                        const float *A, int lda,
                                                        const float *B, int ldb,
                                                        float *C, int ldc) {
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < nr_C; ++i) {
        for (k = 0; k < n_common; ++k) {
            register float A_PART = k1 * A[i * lda + k];
            for (j = 0; j < nc_C; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

// C = k1*A*B.t + C
void math21_matrix_multiply_k1AB_add_k2C_similar_nt_cpu(int nr_C, int nc_C, int n_common, float k1,
                                                        const float *A, int lda,
                                                        const float *B, int ldb,
                                                        float *C, int ldc) {
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < nr_C; ++i) {
        for (j = 0; j < nc_C; ++j) {
            register float sum = 0;
            for (k = 0; k < n_common; ++k) {
                sum += k1 * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

// C = k1*A.t*B + C
void math21_matrix_multiply_k1AB_add_k2C_similar_tn_cpu(int nr_C, int nc_C, int n_common, float k1,
                                                        const float *A, int lda,
                                                        const float *B, int ldb,
                                                        float *C, int ldc) {
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < nr_C; ++i) {
        for (k = 0; k < n_common; ++k) {
            register float A_PART = k1 * A[k * lda + i];
            for (j = 0; j < nc_C; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

// C = k1*A.t*B.t + C
void math21_matrix_multiply_k1AB_add_k2C_similar_tt_cpu(int nr_C, int nc_C, int n_common, float k1,
                                                        const float *A, int lda,
                                                        const float *B, int ldb,
                                                        float *C, int ldc) {
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < nr_C; ++i) {
        for (j = 0; j < nc_C; ++j) {
            register float sum = 0;
            for (k = 0; k < n_common; ++k) {
                sum += k1 * A[k * lda + i] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}


// math21_matrix_multiply_k1AB_add_k2C_similar
// C = k1*(A*B) + k2*C or similar
void math21_matrix_multiply_k1AB_add_k2C_similar_cpu(int ta, int tb, int nr_C, int nc_C, int n_common, float k1,
                                                     const float *A, int lda,
                                                     const float *B, int ldb,
                                                     float k2,
                                                     float *C, int ldc) {
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",ta, tb, M, N, K, k1, lda, ldb, k2, ldc);
    int i, j;
    for (i = 0; i < nr_C; ++i) {
        for (j = 0; j < nc_C; ++j) {
            C[i * ldc + j] *= k2;
        }
    }
    if (!ta && !tb)
        math21_matrix_multiply_k1AB_add_k2C_similar_nn_cpu(nr_C, nc_C, n_common, k1, A, lda, B, ldb, C, ldc);
    else if (ta && !tb)
        math21_matrix_multiply_k1AB_add_k2C_similar_tn_cpu(nr_C, nc_C, n_common, k1, A, lda, B, ldb, C, ldc);
    else if (!ta && tb)
        math21_matrix_multiply_k1AB_add_k2C_similar_nt_cpu(nr_C, nc_C, n_common, k1, A, lda, B, ldb, C, ldc);
    else
        math21_matrix_multiply_k1AB_add_k2C_similar_tt_cpu(nr_C, nc_C, n_common, k1, A, lda, B, ldb, C, ldc);
}

float* math21_matrix_create_random_cpu(int nr, int nc){
    float *m = (float *) math21_vector_create_with_default_value_cpu(nr*nc, 0);
    math21_vector_set_random_cpu(nr * nc, m);
    return m;
}
