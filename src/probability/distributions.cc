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
#include "inner_cc.h"
#include "distributions_c.h"
#include "distributions.h"
#include "ran.h"

namespace math21 {
    /*
* Convolution is the result of adding two different random variables together. See https://web.stanford.edu/class/cs109/reader/7%20Multivariate.pdf
* X, Y 1-dim rv, then Cov[X,Y] := E[(X − E[X])(Y − E[Y])] = E[XY]−E[X]E[Y]
* X n-dim rv with mean mu and covariance matrix Sig, => Sig = E[(X-mu)(X-mu).T] = E[X*X.T] - mu*mu.T
* X n-dim rv with covariance matrix Sig => Sig is symmetric positive semidefinite.
* X n-dim rv, X ~ N(mu, Sig), => Sig spd.
* X n-dim rv, X ~ N(mu, Sig), => there exists a matrix B in R^(n*n) such that if we define Z = B.inv * (X-mu), then Z ~ N(0, I).
the theorem states that any random variable X with a multivariate Gaussian
distribution can be interpreted as the result of applying a linear transformation
(X = BZ + mu) to some collection of n independent standard normal random variables (Z).
     * */

    ////
    void _math21_random_uniform(NumZ &x, think::RandomEngine &ran, NumR a0, NumR b0) {
        MATH21_ASSERT(b0 >= a0);
        NumZ a, b;
        a = xjR2Z_upper(a0);
        b = xjR2Z_lower(b0);
        MATH21_ASSERT(b >= a, "there is no interger in [" << a0 << ", " << b0 << "]");
        NumN i = ran.draw_NumN();
        x = (NumZ) (i % (b - a + 1) + a);
    }

    void math21_random_uniform(NumZ &x, think::RandomEngine &ran, NumR a0, NumR b0) {
        _math21_random_uniform(x, ran, a0, b0);
    }

    void math21_random_uniform(NumN &x, think::RandomEngine &ran, NumR a0, NumR b0) {
        MATH21_ASSERT(b0 >= a0 && b0 > 0);
        NumZ x0;
        _math21_random_uniform(x0, ran, math21_number_not_less_than(a0, 0), b0);
        x = (NumN) x0;
    }

    void math21_random_uniform(NumR &x, think::RandomEngine &ran, NumR a, NumR b) {
        MATH21_ASSERT(b >= a);
        NumR i = ran.draw_0_1();
        x = i * (b - a) + a;
    }

    ////
    void RanUniform::draw(NumR &x) {
        math21_random_uniform(x, engine, a, b);
    }

    void RanUniform::draw(NumN &x) {
        math21_random_uniform(x, engine, a, b);
    }

    void RanUniform::draw(NumZ &x) {
        math21_random_uniform(x, engine, a, b);
    }

    void math21_random_draw(NumN &x, think::Random &ran) {
        ran.draw(x);
    }

    void math21_random_draw(NumZ &x, think::Random &ran) {
        ran.draw(x);
    }

    void math21_random_draw(NumR &x, think::Random &ran) {
        ran.draw(x);
    }

    NumR math21_pr_poisson_probability(NumN n, NumN lambda) {
        NumR value = xjexp(-(NumZ) lambda) * xjpow(lambda, n) / xjfactorial(n);
        return value;
    }

    // binomial(k|n) = s(p^k * (1-p)^(n-k)), with s = n!/(k!(n-k)!) called binomial coefficient.
    //  n trials, p probability of success.
    //  k: the number of successes over the n trials.
    // n>0, 0<=k<=n
    NumR math21_pr_binomial(NumN n, NumR p, NumN k) {
        return xjfactorial(n) * xjpow(p, k) * xjpow(1 - p, n - k) / (xjfactorial(k) * (xjfactorial(n - k)));
    }


// Compute coordinated functions of a symmetric positive semidefinite matrix.
// This class addresses two issues.  Firstly it allows the pseudoinverse,
// the logarithm of the pseudo-determinant, and the rank of the matrix
// to be computed using one call to eigh instead of three.
    NumB _psd(const MatR &M, NumN &rank, MatR &U, NumR &log_pdet) {
        VecR s;
        MatR u;
        math21_operator_eigen_real_sys_ascending(M, s, u);
        const NumR EPS = std::numeric_limits<NumR>::epsilon();
        if (!math21_operator_container_is_larger_number(s, EPS)) {
            MATH21_ASSERT(0, "singular matrix");
            return 0;
        }
        VecR s_pinv;
        s_pinv.setSize(s.size());
        math21_operator_container_divide(1, s, s_pinv);
        math21_operator_container_sqrt_to(s_pinv);
        MatR s_pinv_mat(s.size(), s.size());
        s_pinv_mat = 0;
        math21_operator_matrix_diagonal_set(s_pinv_mat, s_pinv);
        math21_operator_multiply(1, u, s_pinv_mat, U);
        rank = s.size();
        math21_operator_container_log_can_onto(s, s);
        log_pdet = math21_operator_container_sum(s, 1);
        return 1;
    }

    // X -> Y
    template<typename T>
    const Tensor<T> &math21_broadcast_tensor_if_need(const Tensor<T> &X, TensorBroadcast<T> &X_bc, const VecN &d) {
        if (X.isSameSize(d)) {
            return X;
        } else {
            X_bc.set(X, d);
            return X_bc;
        }
    }

    template<typename T, typename T2, typename T3>
    void math21_broadcast_linear_to_C(NumR k1, const Tensor<T> &A, NumR k2, const Tensor<T2> &B, Tensor<T3> &C, NumN debugCode=0) {
        Seqce<VecN> shapes(2);
        VecN d;
        shapes.at(1) = A.shape(d);
        shapes.at(2) = B.shape(d);
        NumB flag = math21_broadcast_is_compatible_in_ele_op(shapes, d);
        MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                << A.log("A") << B.log("B"));
        TensorBroadcast<T> A_bc;
        TensorBroadcast<T> B_bc;
        const Tensor<T> &A_new = math21_broadcast_tensor_if_need(A, A_bc, d);
        const Tensor<T> &B_new = math21_broadcast_tensor_if_need(B, B_bc, d);
        C.setSize(d);
        math21_operator_container_linear(k1, A_new, k2, B_new, C);
    }

    template<typename T>
    void math21_broadcast_add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
        math21_broadcast_linear_to_C(1, A, 1, B, C);
    }

    template<typename T>
    void math21_broadcast_subtract(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
        math21_broadcast_linear_to_C(1, A, -1, B, C);
    }

    void math21_operator_matrix_stacked_mat_shape_split(const VecN &d_standard, VecN &d_pre, VecN &d_mat) {
        math21_operator_container_subcontainer(d_standard, d_pre, 1, -3);
        math21_operator_container_subcontainer(d_standard, d_mat, -2, -1);
    }

    // A -> A_part, i.e., get i-th matrix in stacked matrices.
    template<typename T>
    void math21_broadcast_share_part_using_index_pre_in_stacked_matmul(
            const Tensor<T> &A, const VecN &index_pre,
            const VecN &d_pre, const VecN &d_mat, Tensor<T> &A_part) {
        VecN index_ori;
        math21_broadcast_index_to_original(index_pre, d_pre, index_ori);
        NumN i = math21_operator_number_index_to_num_right(index_ori, d_pre);
        math21_operator_share_part_i_tensor(A, i, d_mat, A_part);
    }

    template<typename T>
    void math21_broadcast_stacked_matmul(NumR s, const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
        if (A.dims() <= 2 && B.dims() <= 2) {
            math21_operator_matrix_mul_with_trans_option(s, A, B, C);
            return;
        }
        MATH21_ASSERT(A.dims() == B.dims(), "dims not equal")
        MATH21_ASSERT(A.isStandard() && B.isStandard(), "Only standard tensor is supported!")

        VecN d_A, d_B, d_A_new, d_B_new, d_C, d_A_standard, d_B_standard;
        A.shape(d_A);
        B.shape(d_B);
        NumB flag = math21_broadcast_is_compatible_in_stacked_matmul(
                d_A, d_B, d_A_new, d_B_new, d_C, d_A_standard, d_B_standard);
        MATH21_ASSERT(flag, "shape not compatible when broadcasting\n"
                << A.log("A") << B.log("B"));

        C.setSize(d_C);
        VecN d_pre_A, d_pre_B, d_pre_C;
        VecN d_mat_A, d_mat_B, d_mat_C;
        math21_operator_matrix_stacked_mat_shape_split(d_A_standard, d_pre_A, d_mat_A);
        math21_operator_matrix_stacked_mat_shape_split(d_B_standard, d_pre_B, d_mat_B);
        math21_operator_matrix_stacked_mat_shape_split(d_C, d_pre_C, d_mat_C);

        Tensor<T> A_part, B_part, C_part;
        VecN index_pre;
        index_pre.setSize(d_pre_C.size());
        index_pre = 1;
        while (1) {
            math21_broadcast_share_part_using_index_pre_in_stacked_matmul(A, index_pre, d_pre_A, d_mat_A, A_part);
            math21_broadcast_share_part_using_index_pre_in_stacked_matmul(B, index_pre, d_pre_B, d_mat_B, B_part);
            math21_broadcast_share_part_using_index_pre_in_stacked_matmul(C, index_pre, d_pre_C, d_mat_C, C_part);

            math21_operator_matrix_mul_with_trans_option(s, A_part, B_part, C_part);
            if (math21_operator_container_increaseNumFromRight(d_pre_C, index_pre) == 0) {
                break;
            }
        }
    }

//        x : ndarray
//            Points at which to evaluate the log of the probability
//            density function
//        mean : ndarray
//            Mean of the distribution
//        prec_U : ndarray
//            A decomposition such that np.dot(prec_U, prec_U.T)
//            is the precision matrix, i.e. inverse of the covariance matrix.
//        log_det_cov : float
//            Logarithm of the determinant of the covariance matrix
//        rank : int
//            Rank of the covariance matrix.
    void _logpdf2(VecR &value, const MatR &X, const VecR &mean, const MatR &prec_U, NumR log_det_cov, NumN rank) {
        // -(n/2)*log(2*pi) - 0.5*log|Sig| -0.5*sum(square((X-mu.t)*U),axis=-1), where U*U.t = Sig^-1
        // dev = x - mean
        // maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        // return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
        MatR mean_trans;
        math21_operator_matrix_trans(mean, mean_trans);
        MatR dev;
        math21_broadcast_subtract(X, mean_trans, dev);

        MatR t1;
        math21_operator_matrix_mul_with_trans_option(1, dev, prec_U, t1);
        math21_operator_container_square_to(t1);

        VecR maha;
        VecZ axes(1);
        axes = -1;
        math21_operator_tensor_sum_along_axes(t1, maha, axes);
        NumN n_data = X.nrows();
        value.setSize(n_data);
        math21_operator_container_linear_kx_b(-0.5, maha, -0.5 * (rank * xjlog(2 * MATH21_PI) + log_det_cov), value);
    }

    NumR _logpdf(const VecR &x, const VecR &mean, const MatR &prec_U, NumR log_det_cov, NumN rank) {
        // dev = x - mean
        // maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        // return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
        VecR dev(x.size());
        math21_operator_container_subtract_to_C(x, mean, dev);
        MatR t1;
        math21_operator_matrix_mul_with_trans_option(1, dev, prec_U, t1, 1);
        math21_operator_container_square_to(t1);
        NumR maha = math21_operator_container_sum(t1, 1);
        return -0.5 * (rank * xjlog(2 * MATH21_PI) + log_det_cov + maha);
    }

    // x -> batch x
    const MatR &_math21_pr_mvn_logpdf_x_vec_to_mat(const MatR &x, MatR &x_new) {
        if (x.dims() == 1) {
            math21_operator_tensor_shallow_copy(x, x_new);
            math21_operator_tensor_add_axis(x_new, 1);
            return x_new;
        } else {
            return x;
        }
    }

    /*
 *
 multivariate normal distribution
 *
 X is an n-dim random variable,
 X ~ N(mu, Sig) <=> a.T*X ~ N(a.T*mu, a.T*Sig*a),
 where mu = E(X), Sig = Cov(X)
 *
 pdf:
 if Sig nonsingular,
 then f(x) = k * exp(-0.5*(x-mu).T*Sig^-1*(x-mu)),
 with k = 1/((2*pi)^(n/2)*|Sig|^0.5)
 else X does not have a density.
 where |Sig| is determinant of matrix Sig
 *
 logpdf: y = logf(x) = -(n/2)*log(2*pi) - 0.5*log|Sig| -0.5*(x-mu).t*Sig^-1*(x-mu)
     dy/dx = -(x-mu).t * Sig^-1
     dy/dmu = (x-mu).t * Sig^-1
     dy/dSig = -1/2*Sig^-1 + 1/2*L*L.t, with L = Sig^-1*(x-mu)
     (See math21_operator_matrix_ad_reverse_inv)

     (1) Y = logf(X) = -(n/2)*log(2*pi) - 0.5*log|Sig| -0.5*sum(square((X-mu.t)*U),axis=-1), where U*U.t = Sig^-1
     (2) Y = logf(X) = -(n/2)*log(2*pi) - 0.5*log|Sig| -0.5*{addaxis2((X-mu).t)*Sig^-1*addaxis3(X-mu)}.toVec
     where X^t = (x1, x2, ..., xb), Y^t = (y1, y2, ..., yb), X has shape: b * n_x
     Note: * can mean elemul, matmul or stacked matmul
 * */
    NumB math21_pr_mvn_logpdf2(const MatR &x0, const VecR &mean, const MatR &covariance, VecR &value) {
        MatR x_new;
        const MatR &x = _math21_pr_mvn_logpdf_x_vec_to_mat(x0, x_new);
        NumN rank;
        MatR U;
        NumR log_pdet;
        if (!_psd(covariance, rank, U, log_pdet)) {
            return 0;
        }
        _logpdf2(value, x, mean, U, log_pdet, rank);
        return 1;
    }

    // deprecate, use above instead.
    NumB math21_pr_mvn_logpdf(const VecR &x, const VecR &mean, const MatR &covariance, NumR &value) {
        NumN rank;
        MatR U;
        NumR log_pdet;
        if (!_psd(covariance, rank, U, log_pdet)) {
            return 0;
        }
        value = _logpdf(x, mean, U, log_pdet, rank);
        return 1;
    }

    // todo: make generic for m21point, ad_point, or MatR
    // compute dY/dX
    // (1) dy/dx = -(x-mu).t * Sig^-1, x shape: n_x
    // (2) dY/dX = diag(V), with V = -(X-mu.t) * Sig^-1, where X^t = (x1, x2, ..., xb), Y^t = (y1, y2, ..., yb), X has shape: b * n_x
    // (3) dL/dX = dL/dY *.m dY/dX = (dL/dY).t *.e V, here meaning of * is obvious.
    void math21_pr_mvn_dYdX_diag_logpdf(const MatR &X0, const VecR &mean, const MatR &covariance, MatR &dx) {
        MatR x_new;
        const MatR &X = _math21_pr_mvn_logpdf_x_vec_to_mat(X0, x_new);

        MatR mean_trans;
        math21_operator_matrix_trans(mean, mean_trans);
        VecR x_mu;
        math21_broadcast_subtract(X, mean_trans, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        math21_operator_matrix_mul_with_trans_option(-1, x_mu, Sig_inv, dx);
    }

    void math21_pr_mvn_dYdX_diag_logpdf_bak(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dx) {
        VecR x_mu(x.size());
        math21_operator_container_subtract_to_C(x, mean, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        math21_operator_matrix_mul_with_trans_option(-1, x_mu, Sig_inv, dx, 1, 0);
    }

    // compute dY/dmu
    // (1) dy/dmu = (x-mu).t * Sig^-1
    // (2) dY/dmu = (X-mu.t) * Sig^-1, where X^t = (x1, x2, ..., xb), Y^t = (y1, y2, ..., yb), X has shape: b * n_x
    // (3) dL/dmu = dL/dY * dY/dmu
    void math21_pr_mvn_dYdmu_logpdf(const MatR &X0, const VecR &mean, const MatR &covariance, MatR &dmu) {
        MatR x_new;
        const MatR &X = _math21_pr_mvn_logpdf_x_vec_to_mat(X0, x_new);

        MatR mean_trans;
        math21_operator_matrix_trans(mean, mean_trans);
        VecR x_mu;
        math21_broadcast_subtract(X, mean_trans, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        math21_operator_matrix_mul_with_trans_option(1, x_mu, Sig_inv, dmu);
    }

    void math21_pr_mvn_dYdmu_logpdf_bak(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dmu) {
        VecR x_mu(x.size());
        math21_operator_container_subtract_to_C(x, mean, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        math21_operator_matrix_mul_with_trans_option(1, x_mu, Sig_inv, dmu, 1, 0);
    }

    // compute dY/dSig
    // (1) dy/dSig = -1/2*Sig^-1 + 1/2*L*L.t, with L = Sig^-1*(x-mu)
    // (2) dY/dSig = (-1/2*Sig^-1).addaxis(1) + 1/2*L*L.t, with L_raw = (X-mu.t)*Sig^-1,
    // L = addaxism1(L_raw), L.t = swapaxes(L, -1, -2), and * in L*L.t is stacked matmul
    // where X^t = (x1, x2, ..., xb), Y^t = (y1, y2, ..., yb), X has shape: b * n_x
    // (3) dL/dSig = dL/dY * dY/dSig
    void math21_pr_mvn_dYdSig_logpdf(const MatR &X0, const VecR &mean, const MatR &covariance, MatR &dSig) {
        MatR x_new;
        const MatR &X = _math21_pr_mvn_logpdf_x_vec_to_mat(X0, x_new);

        MatR mean_trans;
        math21_operator_matrix_trans(mean, mean_trans);
        VecR x_mu;
        math21_broadcast_subtract(X, mean_trans, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        MatR L_raw;
        math21_operator_matrix_mul_with_trans_option(1, x_mu, Sig_inv, L_raw, 0, 0);
        math21_operator_tensor_add_axis(L_raw, -1);
        MatR &L = L_raw;
        MatR L_t;
        // todo: optimize, L^t can be get using L.reshapeTo, so they share data
        math21_op_tensor_swap_axes(L, L_t, -1, -2);
        MatR LLt;
        math21_broadcast_stacked_matmul(0.5, L, L_t, LLt);
        math21_operator_tensor_add_axis(Sig_inv, 1);
        math21_broadcast_linear_to_C(-0.5, Sig_inv, 1, LLt, dSig, 1);
    }

    void math21_pr_mvn_dYdSig_logpdf_bak(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dSig) {
        VecR x_mu(x.size());
        math21_operator_container_subtract_to_C(x, mean, x_mu);
        MatR Sig_inv;
        math21_operator_inverse(covariance, Sig_inv);
        MatR L;
        math21_operator_matrix_mul_with_trans_option(1, Sig_inv, x_mu, L, 0, 0);
        math21_operator_matrix_mul_with_trans_option(0.5, L, L, dSig, 0, 1);
        math21_operator_linear_to_B(-0.5, Sig_inv, 1, dSig);
    }
}

using namespace math21;

void math21_random_draw_c_int(int *x0, void *ran0) {
    think::Random &ran = *(think::Random *) ran0;
    NumZ x;
    math21_random_draw(x, ran);
    *x0 = x;
}

void math21_random_draw_c_float(float *x0, void *ran0) {
    think::Random &ran = *(think::Random *) ran0;
    NumR x;
    math21_random_draw(x, ran);
    *x0 = (float) x;
}
