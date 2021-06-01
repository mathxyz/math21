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

#include <fstream>
#include "files.h"
#include "inner.h"

using namespace math21;
using namespace matlab;

NumR f_ex_x3_x2_x(NumR x) {
    return -0.1 * pow(x, 4) - 0.15 * pow(x, 3) - 0.5 * pow(x, 2) - 0.25 * x + 1.2;
}

NumR f_ex_x3_x2_x_derivative(NumR x) {
    return -0.4 * pow(x, 3) - 0.45 * pow(x, 2) - x - 0.25;
}

NumR f_ex_x3_x2_x_fdm_form(const VecR &x0, const void *data) {
    MATH21_ASSERT(x0.size() == 1)
    return f_ex_x3_x2_x(x0(1));
}

// f
NumR f_ex_cos_sin(NumR a, NumR b, NumR x) {
    return a * cos(b * x) + b * sin(a * x);
}

// f
NumR f_ex_cos_sin_fdm_form(const VecR &x0, const void *pdata) {
    NumR a, b, x;
    a = x0(1);
    b = x0(2);
    const VecR &data = *(const VecR *) pdata;
    x = data(1);
    return a * cos(b * x) + b * sin(a * x);
}

// y has same size with data.
void f_ex_cos_sin_vector_data_batch_fdm_form(const VecR &x0, VecR &y, const void *data) {
    NumR a, b;
    a = x0(1);
    b = x0(2);
    const VecR &x = *(VecR *) data;
    y = a * cos(b * x) + b * sin(a * x);
}

// df/da, df/db
void f_ex_cos_sin_derivative_vector(const VecR &paras, const VecR &x, MatR &df) {
    NumR a, b;
    a = paras(1);
    b = paras(2);
    auto da = cos(b * x) + multiply_elewise(b * x, cos(a * x));
    auto db = -multiply_elewise(a * x, sin(b * x)) + sin(a * x);
    math21_operator_matrix_col_set_by_vec(df, 1, da);
    math21_operator_matrix_col_set_by_vec(df, 2, db);
}

void f_ex_cos_sin_derivative_fdm(const VecR &x, const VecR &data, MatR &df) {
    math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(f_ex_cos_sin_vector_data_batch_fdm_form, x, df, &data);
}

void f_ex_cos_sin_derivative_fdm2(const VecR &x, const VecR &data, MatR &df) {
    VecR data_one(1);
    MatR df_one;
    for (NumN i = 1; i <= data.size(); ++i) {
        data_one = data(i);
        math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(f_ex_cos_sin_vector_data_batch_fdm_form, x, df_one,
                                                                    &data_one);
        if (i == 1) {
            if (!df.isSameSize(data.size() * df_one.nrows(), x.size())) {
                df.setSize(data.size() * df_one.nrows(), x.size());
            }
        }
        math21_operator_matrix_rows_set_with_offsets(df, df_one, df_one.nrows(), (i - 1) * df_one.nrows());
    }
}

void f_ex_x3_x2_x_fdm_test() {
    NumR x = 0.5;
    NumR df = f_ex_x3_x2_x_derivative(x);
    m21log("x", x);
    m21log("df", df);
    m21log("df_fdm_central", math21_fdm_derivative_1_order_2_error_central_diff(f_ex_x3_x2_x, x));
    VecR x_vec(1);
    x_vec = x;
    VecR df_vec(1);
    math21_fdm_derivative_1_order_2_error_central_diff_gradient(
            f_ex_x3_x2_x_fdm_form, x_vec, df_vec, 0);
    df_vec.log("df_fdm_central_vector");
}

// https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/homework/HW5_LM_handout.pdf
// finite difference methods
void f_ex_cos_sin_lm_fdm_test() {
    // test
    NumN isfdm = 1;
//    NumN isfdm = 0;

    // generate the synthetic data with some noise
    VecR paras(2);
    paras = 100, 102;
    paras.log("true paras", 0, 0, 6);
    VecR x;
    math21_matlab_vector_create_with_increment(x, 0, 0.1, 2 * MATH21_PI);
    VecR y;
    f_ex_cos_sin_vector_data_batch_fdm_form(paras, y, &x);
    VecR y_input;
    y_input = y + 5 * rand(x.size(), 1);

    // initial guess for the parameters
    VecR paras_0(2);
    paras_0 = 100.5, 102.5;
    paras_0.log("init paras", 0, 0, 6);

    NumN n_data = y_input.size();
    NumN n_paras = 2;
    NumN n_iters = 100;
    NumR lambda = 0.01;
    NumN updateJ = 1;
    VecR paras_est(2);
    paras_est = paras_0;
    NumN it;

    MatR J(n_data, n_paras);
    MatR H;
    VecR d;
    NumR e;
    timer time;
    time.start();
    for (it = 1; it <= n_iters; ++it) {
        if (updateJ == 1) {
            // evaluate the Jacobian matrix at current paras (a_est, b_est)
            J = 0;
            if (isfdm) {
                f_ex_cos_sin_derivative_fdm(paras_est, x, J);
//                f_ex_cos_sin_derivative_fdm2(paras_est, x, J);
            } else {
                f_ex_cos_sin_derivative_vector(paras_est, x, J);
            }

            // evaluate the distance error at the current parameters
            VecR y_est;
            f_ex_cos_sin_vector_data_batch_fdm_form(paras_est, y_est, &x);
            d = y_input - y_est;
            // compute the approximated Hessian matrix
            H = transpose(J) * J;
            if (it == 1) {
                e = dot(d, d);
            }
        }

        // apply the damping factor to the Hessian matrix
        MatR H_lm = H + (lambda * eye(n_paras, n_paras));

        // compute the updated paras, (J'WJ + lambda*I) * dp = J'W(y_input-y_est)
        MatR dp = inv(H_lm) * (transpose(J) * d);
        VecR paras_lm(2);
        paras_lm = paras_est + dp;

        // evaluate the total distance error at the updated paras.
        VecR y_est_lm;
        f_ex_cos_sin_vector_data_batch_fdm_form(paras_lm, y_est_lm, &x);
        VecR d_lm = y_input - y_est_lm;
        NumR e_lm = dot(d_lm, d_lm);

        // use the updated paras or discard
        if (e_lm < e) {
            lambda = lambda / 10;
            paras_est = paras_lm;
            e = e_lm;
            printf("e: %lf\n", e);
            updateJ = 1;
        } else {
            updateJ = 0;
            lambda = lambda * 10;
        }
    }
    time.end();
    paras_est.log("estimated paras", 0, 0, 6);
    printf("time used %lf ms\n", time.time());
}
