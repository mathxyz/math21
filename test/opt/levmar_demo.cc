///* Copyright 2015 The math21 Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================*/
//
//#include <fstream>
//#include "files.h"
//#include "inner.h"
//
//using namespace math21;
//using namespace matlab;
//
//void f_ex_cos_sin_vector(NumR a, NumR b, const VecR &x, VecR &y) {
//    y = a * cos(b * x) + b * sin(a * x);
//}
//
//// -df
//void f_ex_cos_sin_neg_derivative(NumR a, NumR b, NumR x, NumR &da, NumR &db) {
//    da = -cos(b * x) - x * b * cos(a * x);
//    db = x * a * sin(b * x) - sin(a * x);
//}
//
//static void log_paras(const char *name, NumR a, NumR b) {
//    if (name) {
//        printf("%s\n", name);
//    }
//    printf("a, b =  %lf, %lf\n", a, b);
//}
//
//// https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/homework/HW5_LM_handout.pdf
//void f_ex_cos_sin_lm_test() {
//
//    // generate the synthetic data with some noise
//    NumR a = 100;
//    NumR b = 102;
//    log_paras("true paras", a, b);
//    VecR x;
//    math21_matlab_vector_create_with_increment(x, 0, 0.1, 2 * MATH21_PI);
//    VecR y;
//    f_ex_cos_sin_vector(a, b, x, y);
//    VecR y_input;
//    y_input = y + 5 * rand(x.size(), 1);
//
//    // initial guess for the parameters
//    NumR a0 = 100.5;
//    NumR b0 = 102.5;
//    log_paras("init paras", a0, b0);
//    VecR y_init;
//    f_ex_cos_sin_vector(a0, b0, x, y_init);
//
//    NumN n_data = y_input.size();
//    NumN n_paras = 2;
//    NumN n_iters = 100;
//    NumR lambda = 0.01;
//    NumN updateJ = 1;
//    NumR a_est = a0;
//    NumR b_est = b0;
//    NumN it;
//
//    MatR J(n_data, n_paras);
//    MatR H;
//    VecR d;
//    NumR e;
//    timer time;
//    time.start();
//    for (it = 1; it <= n_iters; ++it) {
//        if (updateJ == 1) {
//            // evaluate the Jacobian matrix at current paras (a_est, b_est)
//            J = 0;
//            for (NumN i = 1; i <= J.nrows(); ++i) {
//                NumR da, db;
//                f_ex_cos_sin_neg_derivative(a_est, b_est, x(i), da, db);
//                J(i, 1) = da;
//                J(i, 2) = db;
//            }
//
//            // evaluate the distance error at the current parameters
//            VecR y_est;
//            f_ex_cos_sin_vector(a_est, b_est, x, y_est);
//            d = y_input - y_est;
//            // compute the approximated Hessian matrix
//            H = transpose(J) * J;
//            if (it == 1) {
//                e = dot(d, d);
//            }
//        }
//
//        // apply the damping factor to the Hessian matrix
//        MatR H_lm = H + (lambda * eye(n_paras, n_paras));
//
//        // compute the updated paras
//        MatR dp = -inv(H_lm) * (transpose(J) * d);
//        NumR a_lm = a_est + dp(1);
//        NumR b_lm = b_est + dp(2);
//
//        // evaluate the total distance error at the updated paras.
//        VecR y_est_lm;
//        f_ex_cos_sin_vector(a_lm, b_lm, x, y_est_lm);
//        VecR d_lm = y_input - y_est_lm;
//        NumR e_lm = dot(d_lm, d_lm);
//
//        // use the updated paras or discard
//        if (e_lm < e) {
//            lambda = lambda / 10;
//            a_est = a_lm;
//            b_est = b_lm;
//            e = e_lm;
//            printf("e: %lf\n", e);
//            updateJ = 1;
//        } else {
//            updateJ = 0;
//            lambda = lambda * 10;
//        }
//    }
//    time.end();
//    log_paras("estimated paras", a_est, b_est);
//    printf("time used %lf ms\n", time.time());
//}
