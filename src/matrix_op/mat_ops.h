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
#include "../probability/data_structure/files.h"

namespace math21 {

    void math21_operator_mat_eye(MatR &A);

    template<typename T>
    void math21_operator_swap_rows(Tensor<T> &A, NumN i, NumN j) {
        MATH21_ASSERT(i <= A.nrows() && j <= A.nrows());
        if (i == j) {
            return;
        }
        NumN n = A.ncols();
        NumN k;
        for (k = 1; k <= n; ++k) {
            m21_swap(A.operator()(i, k), A.operator()(j, k));
        }
    }

    template<typename T>
    void math21_operator_swap_cols(Tensor<T> &A, NumN i, NumN j) {
        MATH21_ASSERT(i <= A.ncols() && j <= A.ncols());
        if (i == j) {
            return;
        }
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            m21_swap(A.operator()(k, i), A.operator()(k, j));
        }
    }

    template<typename T, typename T2>
    void math21_operator_matrix_row_set_by_num(Tensor<T> &A, NumN i, const T2 &x, NumN n = 0, NumN offset = 0) {
        MATH21_ASSERT(i <= A.nrows());
        MATH21_ASSERT(offset < A.ncols(), "offset not in [0, size)");
        NumN n_max = A.ncols() - offset;
        if (n == 0 || n > n_max) {
            n = n_max;
        }
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(i, k + offset) = x;
        }
    }

    // A is Mat, x is Vec
    template<typename T>
    void math21_operator_matrix_row_set_by_vec(Tensor<T> &A, NumN i, const Tensor<T> &x, NumN n = 0,
                                               NumN offset1 = 0, NumN offset2 = 0) {
        MATH21_ASSERT(i <= A.nrows());
        MATH21_ASSERT(offset1 < A.ncols(), "offset not in [0, size)");
        MATH21_ASSERT(offset2 < x.size(), "offset not in [0, size)");
        NumN n_max = xjmin(A.ncols() - offset1, x.size() - offset2);
        if (n == 0 || n > n_max) {
            n = n_max;
        }
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(i, k + offset1) = x(k + offset2);
        }
    }

    // A is Mat, B is Mat
    //   A(1 + offset1 : n + offset1, :)
    // = B(1 + offset2 : n + offset2, :)
    template<typename T1, typename T2>
    void
    math21_operator_matrix_rows_set_with_offsets(Tensor<T1> &A, const Tensor<T2> &B, NumN num = 0,
                                                 NumN ir_offset1 = 0, NumN ir_offset2 = 0) {
        if (num == 0) {
            num = B.nrows();
        }
        MATH21_ASSERT(ir_offset1 + num <= A.nrows());
        MATH21_ASSERT(ir_offset2 + num <= B.nrows());
        MATH21_ASSERT(A.ncols() == B.ncols());
        NumN n = A.ncols();
        NumN k;
        for (NumN ir = 1; ir <= num; ++ir) {
            for (k = 1; k <= n; ++k) {
                A(ir + ir_offset1, k) = B(ir + ir_offset2, k);
            }
        }
    }

    template<typename T>
    void math21_operator_matrix_row_get(const Tensor<T> &A, NumN i, Tensor<T> &x) {
        MATH21_ASSERT(i <= A.nrows());
        MATH21_ASSERT(x.isEmpty() || x.size() == A.ncols());
        if (x.isEmpty()) {
            x.setSize(A.ncols());
        }
        NumN n = A.ncols();
        NumN k;
        for (k = 1; k <= n; ++k) {
            x(k) = A(i, k);
        }
    }

    // set row i in mat A by row j in mat B
    template<typename T>
    void math21_operator_matrix_row_set_by_mat_row(Tensor<T> &A, NumN i, const Tensor<T> &B, NumN j) {
        MATH21_ASSERT(i <= A.nrows() && A.ncols() == B.ncols() && j <= B.nrows());
        NumN n = A.ncols();
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(i, k) = B(j, k);
        }
    }

    // set col i in mat A by col j in mat B
    template<typename T>
    void math21_operator_matrix_col_set_by_mat_col(Tensor<T> &A, NumN i, const Tensor<T> &B, NumN j) {
        MATH21_ASSERT(i <= A.ncols() && A.nrows() == B.nrows() && j <= B.ncols());
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(k, i) = B(k, j);
        }
    }

    template<typename T>
    void math21_operator_matrix_col_set_by_num(Tensor<T> &A, NumN i, T x) {
        MATH21_ASSERT(i <= A.ncols());
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(k, i) = x;
        }
    }

    template<typename T>
    void math21_operator_matrix_col_get(const Tensor<T> &A, NumN i, Tensor<T> &x) {
        MATH21_ASSERT(i <= A.ncols());
        MATH21_ASSERT(x.isEmpty() || x.size() == A.nrows());
        if (x.isEmpty()) {
            x.setSize(A.nrows());
        }
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            x(k) = A(k, i);
        }
    }

    // x can be matrix, but is seen as vector.
    template<typename T>
    void math21_operator_matrix_col_set_by_vec(Tensor<T> &A, NumN i, const Tensor<T> &x) {
        MATH21_ASSERT(i <= A.ncols() && x.size() == A.nrows());
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(k, i) = x(k);
        }
    }

    // todo: may use offset instead of from,
    // todo: A(1 + ir_offset1: nr + ir_offset1, 1 + ic_offset1: nc + ic_offset1)
    //     = B(1 + ir_offset2: nr + ir_offset2, 1 + ic_offset2: nc + ic_offset2)
    // A is Mat, B is Mat
    // nr <= B.nr, nc <= B.nc
    template<typename T>
    void math21_operator_matrix_set_using_matrix(Tensor<T> &A, const Tensor<T> &B,
                                                 NumN nr = 0, NumN nc = 0,
                                                 NumZ ir_from1 = 0, NumZ ic_from1 = 0,
                                                 NumZ ir_from2 = 0, NumZ ic_from2 = 0) {
        math21_number_get_from_only_with_check(A.nrows(), ir_from1);
        math21_number_get_from_and_num_with_check(B.nrows(), ir_from2, nr);
        math21_number_get_from_only_with_check(A.ncols(), ic_from1);
        math21_number_get_from_and_num_with_check(B.ncols(), ic_from2, nc);

        MATH21_ASSERT(ir_from1 + nr <= 1 + A.nrows());
        MATH21_ASSERT(ic_from1 + nc <= 1 + A.ncols());
        for (NumN ir = 1; ir <= nr; ++ir) {
            for (NumN ic = 1; ic <= nc; ++ic) {
                A(ir - 1 + ir_from1, ic - 1 + ic_from1) = B(ir - 1 + ir_from2, ic - 1 + ic_from2);
            }
        }
    }

    // A is Mat, b is Vec, B = A + v
    template<typename T>
    void math21_operator_matrix_add_row_vector_to_B(const Tensor<T> &A, const Tensor<T> &v, Tensor<T> &B) {
        MATH21_ASSERT(v.size() == A.ncols());
        if (!B.isSameSize(A.shape())) {
            B.setSize(A.shape());
        }
        for (NumN ir = 1; ir <= A.nrows(); ++ir) {
            for (NumN ic = 1; ic <= A.ncols(); ++ic) {
                B(ir, ic) = A(ir, ic) + v(ic);
            }
        }
    }

// A is Mat, b is Vec, A <- A + v
    template<typename T>
    void math21_operator_matrix_add_row_vector_to_A(Tensor<T> &A, const Tensor<T> &v) {
        math21_operator_matrix_add_row_vector_to_B(A, v, A);
    }

    // x is type Seqce<Vec>
    template<typename T>
    NumN math21_operator_container2d_max_cols(const Seqce<Tensor<T>> &x) {
        NumN nc_max = 0;
        for (NumN i = 1; i <= x.size(); ++i) {
            const auto &v = x(i);
            if (v.size() > nc_max) {
                nc_max = v.size();
            }
        }
        return nc_max;
    }

    // x is type Seqce<Vec>, y is Mat
    template<typename T, typename T2>
    void math21_operator_container2d_to_matrix(const Seqce<Tensor<T>> &x, Tensor<T> &y, const T2 &bgEle) {
        NumN nr = x.size();
        NumN nc = math21_operator_container2d_max_cols(x);
        y.setSize(nr, nc);
        for (NumN i = 1; i <= nr; ++i) {
            const auto &v = x(i);
            math21_operator_matrix_row_set_by_vec(y, i, v);
        }
        for (NumN i = 1; i <= nr; ++i) {
            const auto &v = x(i);
            if (v.size() < nc) {
                math21_operator_matrix_row_set_by_num(y, i, bgEle, 0, v.size());
            }
        }
    }

    // ragged matrix to matrix
    template<typename T>
    void math21_operator_ragged_matrix_to_matrix(const Seqce<Tensor<T>> &x, Tensor<T> &y) {

    }

    void math21_operator_tensor_f_elementwise_unary(const TenR &x, TenR &y, NumR (*f)(const NumR &x));

    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2),
                                                     const TenB &mask = TenB());

    void math21_operator_tensor_f_elementwise_binary(const NumR &x1, const TenR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2),
                                                     const TenB &mask = TenB());

    void math21_operator_tensor_f_elementwise_binary(const TenR &x1, const NumR &x2, TenR &y,
                                                     NumR (*f)(const NumR &x1, const NumR &x2),
                                                     const TenB &mask = TenB());

    void math21_operator_tensor_f_elementwise_ternary(const TenR &x1, const TenR &x2, const TenR &x3, TenR &y,
                                                      NumR (*f)(const NumR &x1, const NumR &x2, const NumR &x3));

    template<typename T>
    void math21_operator_tensor_slice(const Tensor<T> &A, Tensor<T> &B, const Seqce<VecN> &X) {
        TensorView<T> tv = A.sliceView(X);
        tv.toTensor(B);
    }

    struct TensorFunction_sum {
        TensorFunction_sum() {
        }

        template<typename T>
        NumR compute(const Tensor<T> &A) {
            return math21_operator_container_sum(A, 1);
        }
    };

    template<typename FunType>
    struct TensorFunction_f {
        const FunType &f;

        TensorFunction_f(const FunType &f) : f(f) {
        }

        template<typename T>
        NumR compute(const Tensor<T> &A) {
            return f(A);
        }
    };

    struct TensorFunction_min {
        TensorFunction_min() {
        }

        template<typename T>
        T compute(const Tensor<T> &A) {
            return math21_operator_container_min(A);
        }
    };

    struct TensorFunction_max {
        TensorFunction_max() {
        }

        template<typename T>
        T compute(const Tensor<T> &A) {
            return math21_operator_container_max(A);
        }
    };

    struct TensorFunction_argmax {
        TensorFunction_argmax() {
        }

        template<typename T>
        NumN compute(const Tensor<T> &A) {
            return math21_operator_container_argmax(A);
        }
    };

    struct TensorFunction_argmax_random {
        think::RandomEngine &engine;

        TensorFunction_argmax_random(think::RandomEngine &engine) : engine(engine) {
        }

        template<typename T>
        NumN compute(const Tensor<T> &A) {
            return math21_operator_container_argmax_random(A, engine);
        }
    };

    struct TensorFunction_argmin {
        TensorFunction_argmin() {
        }

        template<typename T>
        NumN compute(const Tensor<T> &A) {
            return math21_operator_container_argmin(A);
        }
    };

    struct TensorFunction_mean {
        template<typename T>
        NumR compute(const Tensor<T> &A) {
            return math21_operator_container_mean(A);
        }
    };

    template<typename T>
    void math21_operator_tensor_shrink(const Tensor<T> &A, Tensor<T> &B, const VecN &X) {
        TensorView<T> tv = A.shrinkView(X);
        tv.toTensor(B);
    }

    // see tf.reduce_sum
    // X(i) is 1 or 0.
    // 1 variable, 0 parameter.
    template<typename T, typename S, typename TensorFunction>
    void math21_operator_tensor_f_shrink_bak(const Tensor<T> &A, Tensor<S> &B, const VecN &X,
                                             TensorFunction &f, NumB isKeepingDims = 0) {
        MATH21_ASSERT(X.size() == A.dims())

        // get parameter shape
        VecN dp;
        math21_operator_tensor_shrink_shape(A, X, dp);

        if (dp.isEmpty()) {
            B.setSize(1);
            B(1) = f.compute(A);
            if (isKeepingDims) {
                VecN d(A.dims());
                d = 1;
                B.reshape(d);
            }
        } else {
            VecN b(X.size());
            b = 1;
            // b = 1 - X, because 0 denotes var in shrinkView
            math21_operator_container_linear_to_A(1, b, -1, X);

            MATH21_ASSERT(dp.size() != A.dims(), "No variable specified!")
            B.setSize(dp);

            VecN index(dp.size());
            index = 1;
            VecN y(b.size());
            while (1) {
                math21_operator_container_replace_inc(b, y, index, (NumN) 1);
                TensorView<T> A_prime = A.shrinkView(y);
                B(index) = f.compute(A_prime);
                if (math21_operator_container_increaseNumFromRight(dp, index) == 0) {
                    break;
                }
            }
            if (isKeepingDims) {
                VecN d(A.dims());
                math21_operator_container_replace_inc(X, d, dp, (NumN) 0);
                B.reshape(d);
            }
        }
    }

    void math21_operator_tensor_f_shrink_axes_to_index(NumN dims, const VecN &axes, VecN &index);

    // see tf.reduce_sum
    // X(i) is 1 or 0.
    // 1 variable, 0 parameter.
    template<typename T, typename S, typename TensorFunction>
    void math21_operator_tensor_f_shrink(const Tensor<T> &A, Tensor<S> &B, const VecN &X,
                                         TensorFunction &f, NumB isKeepingDims = 0) {
        MATH21_ASSERT(X.size() == A.dims())

        // get parameter shape
        VecN dp;
        math21_operator_tensor_shrink_shape(A, X, dp);
        MATH21_ASSERT(dp.size() != A.dims(), "No variable specified!")

        VecN b;
        b.setSize(X.size());
        b = 1;
        // b = 1 - X, because 0 denotes var in shrinkView
        math21_operator_container_linear_to_A(1, b, -1, X);

        VecN index;
        VecN y;
        if (dp.isEmpty()) {
            B.setSize(1);
        } else {
            B.setSize(dp);
            index.setSize(dp.size());
            index = 1;
            y.setSize(b.size());
        }
        NumN np = B.size();
        while (1) {
            if (np == 1) {
                B(1) = f.compute(A);
                break;
            } else {
                math21_operator_container_replace_inc(b, y, index, (NumN) 1);
                TensorView<T> A_prime = A.shrinkView(y);
                B(index) = f.compute(A_prime);
                if (math21_operator_container_increaseNumFromRight(dp, index) == 0) {
                    break;
                }
            }
        }
        if (isKeepingDims) {
            VecN d(A.dims());
            math21_operator_container_replace_inc(X, d, dp, (NumN) 0);
            B.reshape(d);
        }
    }

    template<typename T>
    void
    math21_operator_matrix_submatrix(const Tensor<T> &A, Tensor<T> &B, const VecN &row_indexes,
                                     const VecN &col_indexes) {
        MATH21_ASSERT(A.dims() == 2)
        Seqce<VecN> X;
        X.setSize(2);
        X(1).copyFrom(row_indexes);
        X(2).copyFrom(col_indexes);
        math21_operator_tensor_slice(A, B, X);
    }

    // delete i-th row. i= 1, ..., nrow.
    template<typename T>
    void math21_operator_matrix_delete_row(const Tensor<T> &A, Tensor<T> &B, NumN i) {
        if (A.isEmpty()) {
            return;
        }
        MATH21_ASSERT(i <= A.nrows() && i >= 1);
        if (A.dims() == 1) {
            if (B.isSameSize(A.nrows() - 1) == 0) {
                B.setSize(A.nrows() - 1);
            }
        } else if (B.isSameSize(A.nrows() - 1, A.ncols()) == 0) {
            B.setSize(A.nrows() - 1, A.ncols());
        }
        NumN n = A.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            if (k < i) {
                math21_operator_matrix_row_set_by_mat_row(B, k, A, k);
            } else if (k == i) {
            } else {
                math21_operator_matrix_row_set_by_mat_row(B, k - 1, A, k);
            }
        }
    }

    // delete i-th col. i= 1, ..., ncol.
    template<typename T>
    void math21_operator_matrix_delete_col(const Tensor<T> &A, Tensor<T> &B, NumN i) {
        if (A.isEmpty()) {
            return;
        }
        MATH21_ASSERT(i <= A.ncols() && i >= 1);
        if (B.isSameSize(A.nrows(), A.ncols() - 1) == 0) {
            B.setSize(A.nrows(), A.ncols() - 1);
        }
        NumN n = A.ncols();
        NumN k;
        for (k = 1; k <= n; ++k) {
            if (k < i) {
                math21_operator_matrix_col_set_by_mat_col(B, k, A, k);
            } else if (k == i) {
            } else {
                math21_operator_matrix_col_set_by_mat_col(B, k - 1, A, k);
            }
        }
    }

    // insert new row to i-th row. i= 1, ..., nrow, nrow+1.
    template<typename T>
    void math21_operator_matrix_insert_row_value(const Tensor<T> &A, Tensor<T> &B, NumN i, T x) {
        if (A.isEmpty()) {
            return;
        }
        MATH21_ASSERT(i <= A.nrows() + 1 && i >= 1);
        if (A.dims() == 1) {
            if (B.isSameSize(A.nrows() + 1) == 0) {
                B.setSize(A.nrows() + 1);
            }
        } else if (B.isSameSize(A.nrows() + 1, A.ncols()) == 0) {
            B.setSize(A.nrows() + 1, A.ncols());
        }
        NumN n = B.nrows();
        NumN k;
        for (k = 1; k <= n; ++k) {
            if (k < i) {
                math21_operator_matrix_row_set_by_mat_row(B, k, A, k);
            } else if (k == i) {
                math21_operator_matrix_row_set_by_num(B, k, x);
            } else {
                math21_operator_matrix_row_set_by_mat_row(B, k, A, k - 1);
            }
        }
    }

    // insert new row to i-th row. i= 1, ..., nrow, nrow+1.
    template<typename T>
    void math21_operator_vec_insert_value(const Tensor<T> &A, Tensor<T> &B, NumN i, T x) {
        if (A.isEmpty()) {
            MATH21_ASSERT(i == 1)
            B.setSize(1);
            B = x;
            return;
        }
        math21_operator_matrix_insert_row_value(A, B, i, x);
    }

    // insert new col to i-th col. i= 1, ..., ncol, ncol+1.
    template<typename T>
    void math21_operator_matrix_insert_col_value(const Tensor<T> &A, Tensor<T> &B, NumN i, T x) {
        if (A.isEmpty()) {
            return;
        }
        MATH21_ASSERT(i <= A.ncols() + 1 && i >= 1);
        if (B.isSameSize(A.nrows(), A.ncols() + 1) == 0) {
            B.setSize(A.nrows(), A.ncols() + 1);
        }
        NumN n = B.ncols();
        NumN k;
        for (k = 1; k <= n; ++k) {
            if (k < i) {
                math21_operator_matrix_col_set_by_mat_col(B, k, A, k);
            } else if (k == i) {
                math21_operator_matrix_col_set_by_num(B, k, x);
            } else {
                math21_operator_matrix_col_set_by_mat_col(B, k, A, k - 1);
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_diagonal_set(Tensor<T> &A, const Tensor<T> &x) {
        NumN n = xjmin(A.nrows(), A.ncols());
        MATH21_ASSERT(x.size() == n);
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(k, k) = x(k);
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_diagonal_get(const Tensor<T> &A, Tensor<T> &x) {
        NumN n = xjmin(A.nrows(), A.ncols());
        if (!x.isSameSize(n)) {
            x.setSize(n);
        }
        NumN k;
        for (k = 1; k <= n; ++k) {
            x(k) = A(k, k);
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_diagonal_set_value(Tensor<T> &A, NumR x) {
        NumN n = xjmin(A.nrows(), A.ncols());
        NumN k;
        for (k = 1; k <= n; ++k) {
            A(k, k) = x;
        }
    }


    // A is Mat
    template<typename T>
    T math21_operator_matrix_sum_row_i(const Tensor<T> &A, NumN i) {
        MATH21_ASSERT(!A.isEmpty());
        NumN n = A.ncols();
        NumN k;
        T x = 0;
        for (k = 1; k <= n; ++k) {
            x += A(i, k);
        }
        return x;
    }

    // A is Mat
    template<typename T>
    T math21_operator_matrix_sum_col_i(const Tensor<T> &A, NumN i) {
        MATH21_ASSERT(!A.isEmpty());
        NumN n = A.nrows();
        NumN k;
        T x = 0;
        for (k = 1; k <= n; ++k) {
            x += A(k, i);
        }
        return x;
    }

    // A is Mat
    template<typename T>
    T math21_operator_matrix_trace(const Tensor<T> &A) {
        MATH21_ASSERT(!A.isEmpty());
        NumN n = xjmin(A.nrows(), A.ncols());
        NumN k;
        T x = 0;
        for (k = 1; k <= n; ++k) {
            x += A(k, k);
        }
        return x;
    }

    // A is Mat
    template<typename T>
    T math21_operator_matrix_reverse_trace(const Tensor<T> &A) {
        MATH21_ASSERT(!A.isEmpty());
        MATH21_ASSERT(A.nrows() == A.ncols());
        NumN n = A.nrows();
        NumN k;
        T x = 0;
        for (k = 1; k <= n; ++k) {
            x += A(k, n + 1 - k);
        }
        return x;
    }

    // A.transpose
    template<typename T>
    void math21_operator_matrix_trans(Tensor<T> &A) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        MATH21_ASSERT(A.nrows() == A.ncols());
        NumN n, m;
        n = A.ncols();
        m = n;
        NumN i, j;
        for (i = 1; i <= n; i++) {
            for (j = i + 1; j <= m; j++) {
                m21_swap(A.operator()(i, j), A.operator()(j, i));
            }
        }
    }

    // B = A.transpose
    template<typename T>
    void math21_operator_matrix_trans(const Tensor<T> &A, Tensor<T> &B) {
        MATH21_ASSERT(!A.isEmpty(), "empty matrix");
        if (!B.isSameSize(A.ncols(), A.nrows())) {
            B.setSize(A.ncols(), A.nrows());
        }
        NumN nr, nc;
        NumN i, j;
        nr = B.nrows();
        nc = B.ncols();
        for (i = 1; i <= nr; ++i) {
            for (j = 1; j <= nc; ++j) {
                B(i, j) = A(j, i);
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_x_axis(Tensor<T> &A) {
        MATH21_ASSERT(!A.isEmpty());

        NumN nr = A.nrows();
        NumN nc = A.ncols();
        NumN j;

        NumN n = nr;
        NumN n2 = n / 2;
        for (j = 1; j <= nc; ++j) {
            for (NumN k = 1; k <= n2; ++k) {
                m21_swap(A(k, j), A(n + 1 - k, j));
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_x_axis(const Tensor<T> &A, Tensor<T> &B) {
        MATH21_ASSERT(!A.isEmpty());
        if (!B.isSameSize(A.shape())) {
            B.setSize(A.shape());
        }
        NumN nr = A.nrows();
        NumN nc = A.ncols();
        NumN i, j;
        for (i = 1; i <= nr; ++i) {
            for (j = 1; j <= nc; ++j) {
                B(i, j) = A(nr + 1 - i, j);
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_y_axis(Tensor<T> &A) {
        MATH21_ASSERT(!A.isEmpty());
        NumN nr = A.nrows();
        NumN nc = A.ncols();
        NumN j;

        NumN n = nc;
        NumN n2 = n / 2;
        for (j = 1; j <= nr; ++j) {
            for (NumN k = 1; k <= n2; ++k) {
                m21_swap(A(j, k), A(j, n + 1 - k));
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_y_axis(const Tensor<T> &A, Tensor<T> &B) {
        MATH21_ASSERT(!A.isEmpty());
        if (!B.isSameSize(A.shape())) {
            B.setSize(A.shape());
        }
        NumN nr = A.nrows();
        NumN nc = A.ncols();
        NumN i, j;
        for (i = 1; i <= nr; ++i) {
            for (j = 1; j <= nc; ++j) {
                B(i, j) = A(i, nc + 1 - j);
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_x_axis_then_trans(const Tensor<T> &A, Tensor<T> &B) {
        MATH21_ASSERT(!A.isEmpty());
        if (!B.isSameSize(A.ncols(), A.nrows())) {
            B.setSize(A.ncols(), A.nrows());
        }
        NumN nr, nc;
        NumN i, j;
        nr = B.nrows();
        nc = B.ncols();
        for (i = 1; i <= nr; ++i) {
            for (j = 1; j <= nc; ++j) {
                B(i, j) = A(nc + 1 - j, i);
            }
        }
    }

    // A is Mat
    template<typename T>
    void math21_operator_matrix_reverse_y_axis_then_trans(const Tensor<T> &A, Tensor<T> &B) {
        MATH21_ASSERT(!A.isEmpty());
        if (!B.isSameSize(A.ncols(), A.nrows())) {
            B.setSize(A.ncols(), A.nrows());
        }
        NumN nr, nc;
        NumN i, j;
        nr = B.nrows();
        nc = B.ncols();
        for (i = 1; i <= nr; ++i) {
            for (j = 1; j <= nc; ++j) {
                B(i, j) = A(j, nr + 1 - i);
            }
        }
    }

    template<typename T>
    void math21_operator_matrix_axis_to_image(const Tensor<T> &A, Tensor<T> &B) {
        math21_operator_matrix_reverse_y_axis_then_trans(A, B);
    }

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(const MatR &A, MatR &B);

    // B = inverse of A
    void math21_operator_matrix_2_2_inverse(MatR &A);

    NumR math21_operator_matrix_compute_minor(const MatR &A, NumN i, NumN j);

    NumR math21_operator_matrix_compute_cofactor(const MatR &A, NumN i, NumN j);

    void math21_operator_matrix_compute_cofactor(const MatR &A, MatR &cofactor);

    void math21_operator_matrix_compute_classical_adjoint(const MatR &A, MatR &X);

    NumR math21_operator_matrix_compute_det(const MatR &A);

    void math21_operator_matrix_compute_adjoint(const MatC &A, MatC &X);

    void math21_operator_matrix_3x3_compute_cofactor(const MatR &A, MatR &cofactor);

    NumR math21_operator_matrix_compute_det_using_cofactor(const MatR &A, const MatR &cofactor);

    void math21_operator_matrix_3x3_symmetric_inverse(const MatR &A, MatR &B);

    void math21_operator_matrix_3x3_inverse(const MatR &A, MatR &B);

    void math21_operator_matrix_nx3_pseudoinverse(const MatR &A, MatR &B);

    void math21_operator_matrix_ad_reverse_add(const MatR &dC, MatR &dA_or_dB);

    void math21_operator_matrix_ad_reverse_mul(const MatR &A, const MatR &B, const MatR &dC, MatR &dA_or_dB, NumN pos);

    void math21_operator_matrix_ad_reverse_inv(const MatR &C, const MatR &dC, MatR &dA);

    void math21_operator_matrix_ad_reverse_det(const MatR &A, const MatR &C, const MatR &dC, MatR &dA);
}