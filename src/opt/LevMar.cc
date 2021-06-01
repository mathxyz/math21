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

#include "../matlab/files.h"
#include "files.h"
#include "LevMar.h"

namespace math21 {
    using namespace matlab;

// https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/homework/HW5_LM_handout.pdf
// the Levenberg Marquardt algorithm with finite difference methods
// fit f(x; theta) using data x, y.
    void math21_opt_levmar_fdm(void *data, const MatR &y, const VecR &theta_0,
                               void (*f)(const VecR &paras, MatR &y, const void *data), VecR &theta_est,
                               NumN max_iters, NumN logLevel) {
//    NumN n_data = y.size();
//    NumN n_paras = theta_0.size();
        NumR lambda = 0.01;
        NumN updateJ = 1;
        theta_est = theta_0;
        NumN it;

        // J(n_data, n_paras);
        MatR J;
        MatR H;
        VecR d;
        NumR e;
        timer time;
        time.start();
        TenR H_lm_inv;
        for (it = 1; it <= max_iters; ++it) {
            if (updateJ == 1) {
                // evaluate the Jacobian matrix at current paras theta_est.
                math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(f, theta_est, J, data);

                // evaluate the distance error at the current parameters
                VecR y_est;
                f(theta_est, y_est, data);
                d = (y - y_est).toVector();
                // compute the approximated Hessian matrix
                // H shape: n_paras * n_paras
                H = transpose(J) * J;
                if (it == 1) {
                    e = dot(d, d);
                }
            }

            // apply the damping factor to the Hessian matrix
            MatR H_lm = H + (lambda * eye(H.nrows(), H.ncols()));

            // compute the updated paras, (J'WJ + lambda*I) * dp = J'W(y-y_est)
            if(!math21_operator_inverse(H_lm, H_lm_inv)){
                break;
            }
            MatR dp = H_lm_inv * (transpose(J) * d);
            VecR theta_lm = theta_est + dp;

            // evaluate the total distance error at the updated paras.
            VecR y_est_lm;
            f(theta_lm, y_est_lm, data);
            VecR d_lm = y - y_est_lm;
            NumR e_lm = dot(d_lm, d_lm);

            // use the updated paras or discard
            if (e_lm < e) {
                lambda = lambda / 10;
                theta_est = theta_lm;
                e = e_lm;
                if (logLevel>10) {
                    printf("error: %lf\n", e);
                }
                updateJ = 1;
            } else {
                updateJ = 0;
                lambda = lambda * 10;
            }
        }
        time.end();
        if (logLevel > 1) {
            // evaluate the distance error at the current parameters
            VecR y_est;
            f(theta_est, y_est, data);
            if(logLevel>3){
                y_est.log("y_est");
            }
            printf("error: %lf\n", e);
            printf("time used %lf ms\n", time.time());
        }
    }
}