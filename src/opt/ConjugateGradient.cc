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

#include "ConjugateGradient.h"
#include "OptAlg.h"

using namespace math21;

//class FunctionAlpha;
class FunctionAlpha : public Function {
private:
    Functional &f;
    VecR &x, &p;
public:
    FunctionAlpha(Functional &_f, VecR &_x, VecR &_p) : f(_f), x(_x), p(_p) {
    }

    virtual ~FunctionAlpha() {}

    NumR valueAt(const NumR &alpha) override {
        VecR x1;
        math21_operator_vec_linear(1, x, alpha, p, x1);
        NumR y = f.valueAt(x1);
        return y;
    }

    NumR derivativeValueAt(const NumR &x) override {
        return 0;
    }
};

ConjugateGradient::ConjugateGradient(Functional &f, const VecR &_x0) : f(f) {
    time = 0;
    time_max = MATH21_OPT_TIME_MAX;

    x0.copyFrom(_x0);
    y0 = f.valueAt(x0);
    g0.copyFrom(f.derivativeValueAt(x0));
    math21_operator_vec_linear(-1, g0, p0);
    tol = 0.001;
}


void ConjugateGradient::solve() {
    while (1) {
        FunctionAlpha f_alpha(f, x0, p0);
        IntervalLocation intervalLocation(f_alpha);
        intervalLocation.solve();
        NumR interval_1, interval_2;
        intervalLocation.getInterval(interval_1, interval_2);
        GoldenSectionSearch goldenSectionSearch(f_alpha, interval_1, interval_2);
        goldenSectionSearch.solve();
        alpha = goldenSectionSearch.getMinimum();

        math21_operator_vec_linear(1, x0, alpha, p0, x1);
        y1 = f.valueAt(x1);
        if (xjabs(y1 - y0) < tol) {
            break;
        }
        const VecR &g1 = f.derivativeValueAt(x1);
        math21_operator_vec_linear(1, g1, -1, g0, g3_tmp);
        beta = math21_operator_InnerProduct(1, g3_tmp, g1)/ (math21_operator_InnerProduct(1, g0, g0) + MATH21_EPS);
//        beta = (g1 - g0).dotProd(g1) / (g0.dotProd(g0) + MATH21_EPS);
        math21_operator_vec_linear(-1, g1, beta, p0, p1);
//        p1 = (-1) * g1 + beta * p0;
        x0.assign(x1);
        y0 = y1;
        g0.assign(g1);
        p0.assign(p1);

        if (time >= time_max) {
            break;
        }
        time++;
//        m21log("time", time);
    }
    x0.log();
}
