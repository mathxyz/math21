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

#include "operations.h"
#include "ten_ops.h"

namespace math21 {

    /* RULES: EVERY METHODS MUST OBEY!!!
     * RULE 1: if method input is A and output is A, then we must guarantee A has the same type.
     * i.e., if A is vector when as input, it will still be vector when as output.
     * */


    void math21_operator_vec_linear(NumR k1, const VecR &A, VecR &C) {
        math21_operator_linear(k1, A, C);
        MATH21_ASSERT_CODE(C.dims() == 1, "C is not vector");
    }


    // We require that C must be vector if both A and B are vectors.
    void math21_operator_vec_linear(NumR k1, const VecR &A, NumR k2, const VecR &B, VecR &C) {
        math21_operator_linear(k1, A, k2, B, C);
        MATH21_ASSERT_CODE(C.dims() == 1, "C is not vector");
    }

    // A = k1*A
    void math21_operator_linear_to(NumR k1, TenR &A) {
        math21_operator_linear(k1, A, A);
    }

    // Note: if A is be vector, then output A will be vector. This is guaranteed.
    // A = k1*A + k2*B
    void math21_operator_linear_to_A(NumR k1, MatR &A, NumR k2, const MatR &B) {
        math21_operator_linear(k1, A, k2, B, A);
    }

    // Note: if B is be vector, then output B will be vector. This is guaranteed.
    // B = k1*A + k2*B
    void math21_operator_linear_to_B(NumR k1, const MatR &A, NumR k2, MatR &B) {
        math21_operator_linear(k1, A, k2, B, B);
    }

    // solve A*X=B;
    NumB math21_operator_solve_linear_equation(const MatR &A, const MatR &B, MatR &X) {
        MatR A_inv;
        A_inv.setSize(A.shape());
        A_inv.assign(A);
        if (!math21_operator_container_isEqual(X.shape(), B.shape())) {
            X.setSize(B.shape());
        }
        X.assign(B);
        numerical_recipes::GaussJordanElimination gje;
        if(!gje.solve(A_inv, X)){
            return 0;
        }
        return 1;
    }

    NumB math21_operator_inverse(const MatR &A, MatR &A_inv) {
        A_inv.setSize(A.shape());
        A_inv.assign(A);

        numerical_recipes::GaussJordanElimination gje;
        return gje.solve(A_inv);
    }

    NumB math21_operator_inverse(MatR &A) {
        numerical_recipes::GaussJordanElimination gje;
        return gje.solve(A);
    }

    int math21_compute_num_threads(int n, int min_n) {
#ifdef MATH21_FLAG_IS_PARALLEL
#ifdef MATH21_FLAG_USE_OPENMP
        int max_tn = n / min_n;
        const int g_ncore = omp_get_num_procs();
        int tn = max_tn > g_ncore ? g_ncore : max_tn;
        if (tn < 1) {
            tn = 1;
        }
        printf("tn:%d\n", tn);
        return tn;
#else
        return 1;
#endif
#else
        return 1;
#endif
    }

    // A = s*A*B, * is matrix multiplication
    void math21_operator_multiply_to_A(NumR s, MatR &A, const MatR &B) {
        MatR C;
        math21_operator_multiply(s, A, B, C);
        A.swap(C);
    }

    // B = s*A*B, * is matrix multiplication
    void math21_operator_multiply_to_B(NumR s, const MatR &A, MatR &B) {
        MatR C;
        math21_operator_multiply(s, A, B, C);
        B.swap(C);
    }
}