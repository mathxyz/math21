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
#include "basic/files_c.h"

namespace math21 {
    /*
     * min f(x)
     *
     * */
    class GoldenSectionSearch : public think::Optimization {
    private:
        Function &f;
        NumR a, b, c, d;
        NumR tao, tol;
        NumR fc, fd;
        NumN time, time_max;
    public:
        GoldenSectionSearch(Function &f, const NumR &_a, const NumR &_b);

        virtual ~GoldenSectionSearch() {}

        void solve();

        NumR getMinimum() { return c; }
    };

    /*
     * find interval location of argmin along positive x axis.
     * */
    class IntervalLocation : public think::Optimization {
    private:
        Function &f;
        NumR b0, b1, b2;
        NumR f0,f1,f2;
        NumN time, time_max;
    public:
        IntervalLocation(Function &f, const NumR _b1=0.001);

        virtual ~IntervalLocation() {}

        void solve();

        void getInterval(NumR &a, NumR &b);
    };

    // central difference with O(h*h)
    // You can use Richardson extrapolation to improve derivative estimates.
    // h is step size
    // df(xi) = (f(xi1) - f(xi_1))/(2h), xi1 = xi + h, xi_1 = xi - h
    NumR math21_fdm_derivative_1_order_2_error_central_diff(NumR (*f)(NumR x), NumR xi, NumR h = 0.25);

    void
    math21_fdm_derivative_1_order_2_error_central_diff_gradient(NumR (*f)(const VecR &x, const void *data), VecR &x,
                                                                VecR &df,
                                                                const void *data);

    // y.size = data.size * f.y.size
    // y is vector, or matrix.
    // if y is matrix, y is seen as vector.
    void
    math21_fdm_derivative_1_order_2_error_central_diff_Jacobian(void (*f)(const VecR &x, MatR &y, const void *data),
                                                                const VecR &x0,
                                                                MatR &df,
                                                                const void *data);

    void math21_opt_fit(OptUpdate &update);
}