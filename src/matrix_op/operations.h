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

namespace math21 {

    void math21_operator_vec_linear(NumR k1, const VecR &A, VecR &C);


    void math21_operator_vec_linear(NumR k1, const VecR &A, NumR k2, const VecR &B, VecR &C);

    // A = k1*A
    void math21_operator_linear_to(NumR k1, TenR &A);

    // A = k1*A + k2*B
    void math21_operator_linear_to_A(NumR k1, MatR &A, NumR k2, const MatR &B);

    // B = k1*A + k2*B
    void math21_operator_linear_to_B(NumR k1, const MatR &A, NumR k2, MatR &B);

    // solve A*X=B;
    NumB math21_operator_solve_linear_equation(const MatR &A, const MatR &B, MatR &X);

    NumB math21_operator_inverse(const MatR &A, MatR &A_inv);

    NumB math21_operator_inverse(MatR &A);

    int math21_compute_num_threads(int n, int min_n);

    template<typename T>
    void math21_c_matrix_multiply_no_parallel(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.nrows() == A.ncols(), "matrix size doesn't match in *");
        MATH21_ASSERT(A.isContinuous() && !A.isColumnMajor());
        MATH21_ASSERT(B.isContinuous() && !B.isColumnMajor());

        NumN n, m, r;
        n = A.nrows();
        m = B.ncols();
        r = A.ncols();
        if (C.nrows() != n || C.ncols() != m) {
            if (m == 1) {
                C.setSize(n);
            } else {
                C.setSize(n, m);
            }
        }
        MATH21_ASSERT(C.isContinuous() && !C.isColumnMajor());

        NumN i, j, k;
        const T *A_data = math21_memory_tensor_data_address(A);
        const T *B_data = math21_memory_tensor_data_address(B);
        T *C_data = math21_memory_tensor_data_address(C);
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                NumR sum = 0;
                for (k = 1; k <= r; k++) sum += A_data[(i - 1) * r + (k - 1)] * B_data[(k - 1) * m + (j - 1)];
                C_data[(i - 1) * m + (j - 1)] = (T) s * sum;
            }
        }
    }

    template<typename T>
    void math21_matrix_multiply_kAB(NumR s, const T *A_data, const T *B_data, T *C_data,
                                    NumN nr, NumN nc, NumN n_common) {
        NumN i, j, k;
#pragma omp parallel for private(j, k) collapse(2)
        for (i = 1; i <= nr; i++) {
            for (j = 1; j <= nc; j++) {
                NumR sum = 0;
                for (k = 1; k <= n_common; k++)
                    sum += A_data[(i - 1) * n_common + (k - 1)] * B_data[(k - 1) * nc + (j - 1)];
                C_data[(i - 1) * nc + (j - 1)] = (T) s * sum;
            }
        }
    }

    // MATH21_ASSERT(A.isContinuous() && !A.isColumnMajor());
    // MATH21_ASSERT(B.isContinuous() && !B.isColumnMajor());
    template<typename T>
    void math21_c_matrix_multiply(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.nrows() == A.ncols(), "matrix size doesn't match in *" << A.log("A") << B.log("B"));

        NumN nr, nc, n_common;
        nr = A.nrows();
        nc = B.ncols();
        n_common = A.ncols();
        if (C.nrows() != nr || C.ncols() != nc) {
            if (nc == 1) {
                C.setSize(nr);
            } else {
                C.setSize(nr, nc);
            }
        }
        MATH21_ASSERT(C.isContinuous() && !C.isColumnMajor());
        math21_matrix_multiply_kAB(s, A.getDataAddress(), B.getDataAddress(), C.getDataAddress(),
                                   nr, nc, n_common);
    }

    template<typename T>
    void math21_operator_tensor_matrix_multiply_setSize(const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        // not check n_common shape, just check n_common.
        MATH21_ASSERT(B.nrows_generalized() == A.ncols_generalized(), "matrix size doesn't match in *");
        VecN dr, dc;
        A.row_shape(dr);
        B.col_shape(dc);
        C.setSize(dr, dc);
    }

    template<typename T>
    void
    math21_operator_tensor_matrix_multiply_no_setSize(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        // not check n_common shape, just check n_common.
        MATH21_ASSERT(B.nrows_generalized() == A.ncols_generalized(), "matrix size doesn't match in *");
        MATH21_ASSERT(C.nrows_generalized() == A.nrows_generalized(), "matrix size doesn't match in *");
        MATH21_ASSERT(C.ncols_generalized() == B.ncols_generalized(), "matrix size doesn't match in *");

        if (A.isContinuous() && !A.isColumnMajor()
            && B.isContinuous() && !B.isColumnMajor()
            && C.isContinuous() && !C.isColumnMajor()) {
            math21_matrix_multiply_kAB(s, A.getDataAddress(), B.getDataAddress(), C.getDataAddress(),
                                       A.nrows_generalized(),
                                       B.ncols_generalized(),
                                       B.nrows_generalized());
        } else {
            MATH21_ASSERT(0);
        }
    }

    template<typename T>
    void math21_operator_tensor_matrix_trans_setSize(const Tensor <T> &A, Tensor <T> &B) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        VecN dr, dc;
        A.row_shape(dr);
        A.col_shape(dc);
        B.setSize(dc, dr);
    }

    // C will be vector type if C is n*1 shape.
    // C = s*A*B, * is matrix multiplication
    // data doesn't need to be continous.
    template<typename T>
    void math21_operator_matrix_multiply_general(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.nrows() == A.ncols(), "matrix size doesn't match in *");

        NumN n, m, r;
        n = A.nrows();
        m = B.ncols();
        r = A.ncols();
        if (C.nrows() != n || C.ncols() != m) {
            if (m == 1) {
                C.setSize(n);
            } else {
                C.setSize(n, m);
            }
        }
        NumN i, j, k;
#pragma omp parallel for private(j, k) collapse(2) num_threads(math21_compute_num_threads(n*m, 1))
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                NumR sum = 0;
                for (k = 1; k <= r; k++) sum += A(i, k) * B(k, j);
                C(i, j) = (T) s * sum;
            }
        }
    }

    // C = s*A*B, * is matrix multiplication
    template<typename T>
    void math21_operator_matrix_multiply_general_no_parallel(NumR s, const Tensor <T> &A, const Tensor <T> &B,
                                                             Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.nrows() == A.ncols(), "matrix size doesn't match in *");

        NumN n, m, r;
        n = A.nrows();
        m = B.ncols();
        r = A.ncols();
        if (C.nrows() != n || C.ncols() != m) {
            if (m == 1) {
                C.setSize(n);
            } else {
                C.setSize(n, m);
            }
        }
        NumN i, j, k;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                NumR sum = 0;
                for (k = 1; k <= r; k++) sum += A(i, k) * B(k, j);
                C(i, j) = (T) s * sum;
            }
        }
    }


#ifdef MATH21_FLAG_USE_CUDA
    namespace detail {
        void _math21_c_matrix_multiply_cuda_Num(NumR s, const TenN &A, const TenN &B, TenN &C);

        void _math21_c_matrix_multiply_cuda_Num(NumR s, const TenZ &A, const TenZ &B, TenZ &C);

        void _math21_c_matrix_multiply_cuda_Num(NumR s, const TenR &A, const TenR &B, TenR &C);
    }

    template<typename T>
    void math21_c_matrix_multiply_cuda(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        detail::_math21_c_matrix_multiply_cuda_Num(s, A, B, C);
    }

#endif

    void math21_cuda_test_02();

    template<typename T>
    void math21_operator_multiply(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
#if defined(MATH21_FLAG_USE_CUDA)
        math21_c_matrix_multiply_cuda(s, A, B, C);
#else
        // still use cpu
        if (A.isContinuous() && !A.isColumnMajor()
            && B.isContinuous() && !B.isColumnMajor()) {
            math21_c_matrix_multiply(s, A, B, C);
        } else {
            math21_operator_matrix_multiply_general(s, A, B, C);
        }
#endif
    }

    template<typename T>
    void math21_operator_multiply_no_parallel(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        if (A.isContinuous() && !A.isColumnMajor()
            && B.isContinuous() && !B.isColumnMajor()) {
            math21_c_matrix_multiply_no_parallel(s, A, B, C);
        } else {
            math21_operator_matrix_multiply_general_no_parallel(s, A, B, C);
        }
    }

    // A = s*A*B, * is matrix multiplication
    void math21_operator_multiply_to_A(NumR s, MatR &A, const MatR &B);

    // B = s*A*B, * is matrix multiplication
    void math21_operator_multiply_to_B(NumR s, const MatR &A, MatR &B);

    // C = s*(A.transpose)*B, * is matrix multiplication
    template<typename T>
    void math21_operator_trans_multiply(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.nrows() == A.nrows(), "matrix size doesn't match in *");

        NumN n, m, r;
        n = A.ncols();
        m = B.ncols();
        r = A.nrows();
        if (C.nrows() != n || C.ncols() != m) {
            C.setSize(n, m);
        }

        NumN i, j, k;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                C(i, j) = 0;
                for (k = 1; k <= r; k++) C(i, j) += A(k, i) * B(k, j);
                C(i, j) = s * C(i, j);
            }
        }
    }

    // C = s*A*(B.transpose), * is matrix multiplication
    template<typename T>
    void math21_operator_multiply_trans(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.ncols() == A.ncols(), "matrix size doesn't match in *");

        NumN n, m, r;
        n = A.nrows();
        m = B.nrows();
        r = A.ncols();
        if (C.nrows() != n || C.ncols() != m) {
            C.setSize(n, m);
        }

        NumN i, j, k;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                C(i, j) = 0;
                for (k = 1; k <= r; k++) C(i, j) += A(i, k) * B(j, k);
                C(i, j) = s * C(i, j);
            }
        }
    }

    // C = s*(A.transpose)*(B.transpose), * is matrix multiplication
    template<typename T>
    void math21_operator_trans_multiply_trans(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C) {
        MATH21_ASSERT(!A.isEmpty() && !B.isEmpty(), "empty matrix");
        MATH21_ASSERT(B.ncols() == A.nrows(), "matrix size doesn't match in *");

        NumN n, m, r;
        n = A.ncols();
        m = B.nrows();
        r = A.nrows();
        if (C.nrows() != n || C.ncols() != m) {
            C.setSize(n, m);
        }

        NumN i, j, k;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                C(i, j) = 0;
                for (k = 1; k <= r; k++) C(i, j) += A(k, i) * B(j, k);
                C(i, j) = s * C(i, j);
            }
        }
    }

    // see math21_op_mat_mul
    // slow
    template<typename T>
    void math21_operator_matrix_mul_with_trans_option(NumR s, const Tensor <T> &A, const Tensor <T> &B, Tensor <T> &C,
                                                      NumB isTransA = 0,
                                                      NumB isTransB = 0) {
        if (!isTransA && !isTransB) {
            math21_operator_multiply(s, A, B, C);
        } else if (isTransA && isTransB) {
            math21_operator_trans_multiply_trans(s, A, B, C);
        } else if (isTransA) {
            math21_operator_trans_multiply(s, A, B, C);
        } else {
            math21_operator_multiply_trans(s, A, B, C);
        }
    }

}