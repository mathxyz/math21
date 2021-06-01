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

#include "OptAlg.h"
#include "files.h"

namespace math21 {

    //#######################################
    GoldenSectionSearch::GoldenSectionSearch(Function &f, const NumR &_a, const NumR &_b) : f(f), a(_a), b(_b) {
        tao = 0.618;
        tol = 0.001;
        time = 0;
        time_max = MATH21_TIME_MAX;
        c = a + (1 - tao) * (b - a);
        d = b - (1 - tao) * (b - a);
        fc = f.valueAt(c);
        fd = f.valueAt(d);
    }

    void GoldenSectionSearch::solve() {
        while (1) {
            if (fc < fd) {
//                a = a;
                b = d;
                d = c;
                c = a + (1 - tao) * (b - a);
                fd = fc;
                fc = f.valueAt(c);
            } else {
                a = c;
//                b = b;
                c = d;
                d = b - (1 - tao) * (b - a);
                fc = fd;
                fd = f.valueAt(d);
            }

            if (b - a < tol) {
                break;
            }
        }
    }

    //#####################################
    IntervalLocation::IntervalLocation(Function &f, const NumR _b1) : f(f) {
        b0 = 0;
        b1 = _b1;
        f0 = f.valueAt(b0);
        f1 = f.valueAt(b1);
        time = 0;
        time_max = 30;
    }

    void IntervalLocation::solve() {
        if (f1 > f0) {
            b2 = b1;
            return;
        }
        while (1){
            b2 = 2*b1;
            f2 = f.valueAt(b2);
            if(f2>=f1)return;
            b0 = b1;
            b1 = b2;
            f1 = f2;

            if (time >= time_max) {
                MATH21_ASSERT(0, "IntervalLocation fail");
                break;
            }
            time++;
//            m21log("interval location", time);
//            m21log("interval location", f2);
        }
    }

    void IntervalLocation::getInterval(NumR &a, NumR &b) {
        a = b0;
        b = b2;
    }

    // central difference with O(h*h)
    // You can use Richardson extrapolation to improve derivative estimates.
    // h is step size
    // df(xi) = (f(xi1) - f(xi_1))/(2h), xi1 = xi + h, xi_1 = xi - h
    NumR math21_fdm_derivative_1_order_2_error_central_diff(NumR (*f)(NumR x), NumR xi, NumR h) {
        NumR xi1 = xi + h;
        NumR xi_1 = xi - h;
        return (f(xi1) - f(xi_1)) / (2 * h);
    }

    void
    math21_fdm_derivative_1_order_2_error_central_diff_gradient(NumR (*f)(const VecR &x, const void *data), VecR &x,
                                                                VecR &df,
                                                                const void *data) {
        if (!df.isSameSize(x.size())) {
            df.setSize(x.size());
        }
        NumR h = 0.25;
        NumN i;
        NumN n = x.size();
        for (i = 1; i <= n; ++i) {
            x(i) = x(i) + h;
            NumR yi1 = f(x, data);
            x(i) = x(i) - 2 * h;
            NumR yi_1 = f(x, data);
            x(i) = x(i) + h;
            df(i) = (yi1 - yi_1) / (2 * h);
        }
    }

    // y.size = data.size * f.y.size
    // y is vector, or matrix.
    // if y is matrix, y is seen as vector.
    void
    math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(void (*f)(const VecR &x, MatR &y, const void *data),
                                                                const VecR &x0,
                                                                MatR &df,
                                                                const void *data) {
        NumR h = 0.25;
        NumN i;
        VecR x;
        x = x0;
        NumN n = x.size();
        // yi1, yi_1 can be matrix or vector.
        MatR yi1;
        MatR yi_1;
        TenR v;
        for (i = 1; i <= n; ++i) {
            x(i) = x(i) + h;
            f(x, yi1, data);
            x(i) = x(i) - 2 * h;
            f(x, yi_1, data);
            x(i) = x(i) + h;
            if (i == 1) {
                if (!df.isSameSize(yi1.size(), x.size())) {
                    df.setSize(yi1.size(), x.size());
                }
            }
            // yi1, yi_1 are seen as vectors.
            math21_operator_tensor_f_elementwise_binary(yi1, yi_1, v, xjsubtract);
            math21_operator_container_linear_kx_b_to(1 / (2 * h), v, 0);
            math21_operator_matrix_col_set_by_vec(df, i, v);
        }
    }

    void math21_opt_fit(OptUpdate &update) {
        if (update.type != OptUpdateType_LevMar) {
            m21log("fit only support LevMar algorithm!");
            return;
        }
        auto *levMar = (OptUpdate_LevMar *) update.detail;
        math21_opt_levmar_fdm(levMar->data, *(levMar->y), *(levMar->theta_0),
                              levMar->f, *(levMar->theta_est), levMar->max_iters, levMar->logLevel);
    }
}