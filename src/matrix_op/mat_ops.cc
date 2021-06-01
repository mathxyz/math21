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

#include "../op/files.h"
#include "operations.h"
#include "mat_ops.h"

namespace math21 {

    void math21_operator_mat_eye(MatR &A) {
        if (A.isEmpty()) {
            return;
        }
        A = 0;
        NumN r = m21_min(A.nrows(), A.ncols());
        NumN i;
        for (i = 1; i <= r; i++) A(i, i) = 1;
    }


#ifndef MATH21_FLAG_USE_CUDA

    void math21_cuda_test_02() {}

#endif

    NumB math21_operator_tensor_f_elementwise_is_compatible(const TenR &x1, const TenR &x2) {
        NumN n = xjmin(x1.dims(), x2.dims());
        for (NumN i = 1; i <= n; ++i) {
            if (x1.dim(i) != x2.dim(i) && x1.dim(i) > 1 && x2.dim(i) > 1) {
                return 0;
            }
        }
        return 1;
    }

    void math21_operator_tensor_f_elementwise_compatible_get_shape(const TenR &x1, const TenR &x2, VecN &d) {
        NumN n = xjmax(x1.dims(), x2.dims());
        d.setSize(n);
        NumN d1, d2;
        for (NumN i = 1; i <= n; ++i) {
            d1 = i > x1.dims() ? 1 : x1.dim(i);
            d2 = i > x2.dims() ? 1 : x2.dim(i);
            d(i) = xjmax(d1, d2);
        }
    }

    // see matlab compatible-array-sizes-for-basic-operations
    void math21_operator_tensor_f_elementwise_compatible_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                                NumR (*f)(const NumR &x1, const NumR &x2),
                                                                const TenB &mask) {
        MATH21_ASSERT(math21_operator_tensor_f_elementwise_is_compatible(x1, x2))
        VecN d;
        math21_operator_tensor_f_elementwise_compatible_get_shape(x1, x2, d);
        if (!y.isSameSize(d)) {
            y.setSize(d);
        }
        VecN index;
        index.setSize(d.shape());
        index = 1;
        VecN index1;
        index1.setSize(x1.dims());
        index1 = 1;
        VecN index2;
        index2.setSize(x2.dims());
        index2 = 1;
        NumN n = d.size();
        NumB isUseMask = 0;
        if (mask.isSameSize(y.shape())) {
            isUseMask = 1;
        }
        while (1) {
            for (NumN i = 1; i <= n; ++i) {
                if (i <= x1.dims()) {
                    index1(i) = x1.dim(i) == 1 ? 1 : index(i);
                }
                if (i <= x2.dims()) {
                    index2(i) = x2.dim(i) == 1 ? 1 : index(i);
                }
            }
            if (isUseMask) {
                if (mask(index)) {
                    y(index) = f(x1(index1), x2(index2));
                }
            } else {
                y(index) = f(x1(index1), x2(index2));
            }
            if (math21_operator_container_increaseNumFromRight(d, index) == 0) {
                break;
            }
        }
    }

    void math21_operator_tensor_f_elementwise_unary(const TenR &x, TenR &y, NumR (*f)(const NumR &x)) {
        if (!y.isSameSize(x.shape())) {
            y.setSize(x.shape());
        }
        math21_operator_container_f_elementwise_unary(x, y, f);
    }

    // That x + y is legal when like x has shape (2,2) and y has shape (1,4)
    void math21_operator_tensor_as_container_f_elementwise_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                                  NumR (*f)(const NumR &x1, const NumR &x2),
                                                                  const TenB &mask) {
        if (!y.isSameSize(x1.shape())) {
            y.setSize(x1.shape());
        }
        math21_operator_container_f_elementwise_binary(x1, x2, y, f, mask);
    }

    // That x + y is illegal when like x has shape (2,2) and y has shape (1,4)
    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        if (x1.isSameSize(x2.shape())) {
            math21_operator_tensor_as_container_f_elementwise_binary(x1, x2, y, f, mask);
        } else {
            math21_operator_tensor_f_elementwise_compatible_binary(x1, x2, y, f, mask);
        }
    }

    void math21_operator_tensor_f_elementwise_binary(const NumR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        TenR v(1);
        v = x1;
        math21_operator_tensor_f_elementwise_binary(v, x2, y, f, mask);
    }

    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const NumR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2), const TenB &mask) {
        TenR v(1);
        v = x2;
        math21_operator_tensor_f_elementwise_binary(x1, v, y, f, mask);
    }

    void math21_operator_tensor_f_elementwise_ternary(const TenR &x1, const TenR &x2, const TenR &x3, TenR &y,
                                                      NumR (*f)(const NumR &x1, const NumR &x2, const NumR &x3)) {
        if (!y.isSameSize(x1.shape())) {
            y.setSize(x1.shape());
        }
        math21_operator_container_f_elementwise_ternary(x1, x2, x3, y, f);
    }

    void math21_operator_tensor_f_shrink_axes_to_index(NumN dims, const VecN &axes, VecN &index) {
        index.setSize(dims);
        if (axes.isEmpty()) {
            index = 1;
        } else {
            index = 0;
            math21_tool_assert(axes.size() <= dims);
            for (NumN i = 1; i <= axes.size(); ++i) {
                math21_tool_assert(axes(i) <= index.size());
                index(axes(i)) = 1;
            }
        }
    }

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(const MatR &A, MatR &B) {
        MATH21_ASSERT(A.isSameSize(2, 2))
        if (!B.isSameSize(2, 2)) {
            B.setSize(2, 2);
        }
        NumR a, b, c, d, det;
        det = A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2);
        MATH21_ASSERT(det != 0)
        a = A(2, 2) / det;
        b = -A(2, 1) / det;
        c = -A(1, 2) / det;
        d = A(1, 1) / det;
        B =
                a, c,
                b, d;
    }

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(MatR &A) {
        math21_operator_matrix_2_2_inverse(A, A);
    }

    // Mij = minor of Aij
    NumR math21_operator_matrix_compute_minor(const MatR &A, NumN i, NumN j) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

// Cij = cofactor of Aij, Cij = (-1)^(i+j) * Mij, Mij is minor of Aij
    NumR math21_operator_matrix_compute_cofactor(const MatR &A, NumN i, NumN j) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

// co-factor matrix of A
    void math21_operator_matrix_compute_cofactor(const MatR &A, MatR &cofactor) {
        math21_tool_assert(0 && "not implement");
    }

    // A*X = det(A)*I = X*A
    // X is classical adjoint (also called adjugate) of A.
    // X = cofactor transpose of A
    void math21_operator_matrix_compute_classical_adjoint(const MatR &A, MatR &X) {
        math21_tool_assert(0 && "not implement");
    }

    // determinant, |A|
    NumR math21_operator_matrix_compute_det(const MatR &A) {
        math21_tool_assert(0 && "not implement");
        return 0;
    }

    // adjoint is the conjugate transpose of A
    void math21_operator_matrix_compute_adjoint(const MatC &A, MatC &X) {
        math21_tool_assert(0 && "not implement");
    }

    // see math21_operator_matrix_compute_minor
    // minor matrix of A
    void math21_operator_matrix_3x3_compute_minor(const MatR &A, MatR &minorM) {
        MATH21_ASSERT(A.isSameSize(3, 3));
        if (!minorM.isSameSize(3, 3)) {
            minorM.setSize(3, 3);
        }
        minorM(1, 1) = A(2, 2) * A(3, 3) - A(2, 3) * A(3, 2);
        minorM(1, 2) = A(2, 1) * A(3, 3) - A(2, 3) * A(3, 1);
        minorM(1, 3) = A(2, 1) * A(3, 2) - A(2, 2) * A(3, 1);
        minorM(2, 1) = A(1, 2) * A(3, 3) - A(1, 3) * A(3, 2);
        minorM(2, 2) = A(1, 1) * A(3, 3) - A(1, 3) * A(3, 1);
        minorM(2, 3) = A(1, 1) * A(3, 2) - A(1, 2) * A(3, 1);
        minorM(3, 1) = A(1, 2) * A(2, 3) - A(1, 3) * A(2, 2);
        minorM(3, 2) = A(1, 1) * A(2, 3) - A(1, 3) * A(2, 1);
        minorM(3, 3) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    }

    // see math21_operator_matrix_compute_cofactor
    // co-factor matrix of A,
    // minor is minor of A
    void math21_operator_matrix_compute_cofactor_using_minor(const MatR &minorM, MatR &cofactor) {
        MATH21_ASSERT(minorM.dims() == 2)
        NumN nr, nc, ir, ic;
        nr = minorM.nrows();
        nc = minorM.ncols();
        if (!cofactor.isSameSize(nr, nc)) {
            cofactor.setSize(nr, nc);
        }
        for (ir = 1; ir <= nr; ++ir) {
            for (ic = 1; ic <= nc; ++ic) {
                cofactor(ir, ic) = minorM(ir, ic) * ((ir + ic) % 2 == 0 ? 1 : -1);
            }
        }
    }

    void math21_operator_matrix_compute_cofactor_using_minor(MatR &minor) {
        math21_operator_matrix_compute_cofactor_using_minor(minor, minor);
    }

    void math21_operator_matrix_3x3_compute_cofactor(const MatR &A, MatR &cofactor) {
        math21_operator_matrix_3x3_compute_minor(A, cofactor);
        math21_operator_matrix_compute_cofactor_using_minor(cofactor);
    }

    NumR math21_operator_matrix_compute_det_using_cofactor(const MatR &A, const MatR &cofactor) {
        NumN nc, ic;
        nc = A.ncols();
        NumR det = 0;
        for (ic = 1; ic <= nc; ++ic) {
            det += A(1, ic) * cofactor(1, ic);
        }
        return det;
    }

    // output: inverse of A
    void math21_operator_matrix_3x3_symmetric_inverse(const MatR &A, MatR &B) {
        math21_operator_matrix_3x3_compute_cofactor(A, B);
        NumR det = math21_operator_matrix_compute_det_using_cofactor(A, B);
        MATH21_ASSERT(det != 0, "singular matrix");
        // cofactor is symmetric
        math21_op_vector_kx_onto(1 / det, B);
    }

// output: inverse of A
    void math21_operator_matrix_3x3_inverse(const MatR &A, MatR &B) {
        math21_operator_matrix_3x3_compute_cofactor(A, B);
        NumR det = math21_operator_matrix_compute_det_using_cofactor(A, B);
        MatR adjugate;
        math21_op_matrix_trans(B, adjugate);
        math21_op_mul(1 / det, adjugate, B);
    }

// see math21_operator_matrix_2_2_inverse
// https://mathworld.wolfram.com/Matrix1-Inverse.html
// https://mathworld.wolfram.com/Moore-PenroseMatrixInverse.html
// we use Moore-PenroseMatrixInverse, A: n*3, B: 3*n
// B = (A.t * A).inv * A.t
    void math21_operator_matrix_nx3_pseudoinverse(const MatR &A, MatR &B) {
        MATH21_ASSERT(A.dim(2) == 3);
        MatR AtA;
        math21_op_mat_mul(A, A, AtA, 1, 0);
        MatR inv;
        math21_operator_matrix_3x3_symmetric_inverse(AtA, inv);
        math21_op_mat_mul(inv, A, B, 0, 1);
    }

    // y = f(C), C = A+B, y in R => dy/dA = dy/dC, dy/dB = dy/dC
    void math21_operator_matrix_ad_reverse_add(const MatR &dC, MatR &dA_or_dB) {
        dA_or_dB.copyFrom(dC);
    }

    // y = f(C), C = A*B, y in R => dy/dA = dy/dC * B.t, dy/dB = A.t * dy/dC
    void math21_operator_matrix_ad_reverse_mul(
            const MatR &A, const MatR &B, const MatR &dC, MatR &dA_or_dB, NumN pos) {
        if (pos == 1) {
            math21_operator_matrix_mul_with_trans_option(1, dC, B, dA_or_dB, 0, 1);
        } else {
            math21_operator_matrix_mul_with_trans_option(1, A, dC, dA_or_dB, 1, 0);
        }
    }

    // References:
    // [An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation] by Mike Giles.
    // (https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
    // y = f(C), C = A.inv, y in R => dy/dA = -C.t * dy/dC * C.t
    void math21_operator_matrix_ad_reverse_inv(const MatR &C, const MatR &dC, MatR &dA) {
        MatR tmp;
        math21_operator_matrix_mul_with_trans_option(1, C, dC, tmp, 1, 0);
        math21_operator_matrix_mul_with_trans_option(-1, tmp, C, dA, 0, 1);
    }

    // y = f(C), C = |A|, y in R => dy/dA = dy/dC * C * A^(-T)
    void math21_operator_matrix_ad_reverse_det(const MatR &A, const MatR &C, const MatR &dC, MatR &dA) {
        MatR tmp;
        math21_operator_matrix_mul_with_trans_option(1, dC, C, tmp, 0, 0);
        MatR A_inv;
        math21_operator_inverse(A, A_inv);
        math21_operator_matrix_mul_with_trans_option(1, tmp, A_inv, dA, 0, 1);
    }
}