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

#include "gje.h"
#include "mrq.h"

namespace math21 {
    namespace numerical_recipes {
        NumB Fitmrq::fit(NumB isLog) {
            NumN j, k, l, iter, done = 0;
            NumR alamda = .001, ochisq; // original chisq
            VecR atry(ma), beta(ma), da(ma);
            mfit = 0;
            for (j = 1; j <= ma; j++) if (ia(j)) mfit++;
            MatR oneda(mfit, 1), temp(mfit, mfit);
            mrqcof(a, alpha, beta);
            for (j = 1; j <= ma; j++) atry(j) = a(j);
            ochisq = chisq;
            for (iter = 0; iter < ITMAX; iter++) {
                if (isLog && iter % 10 == 0) {
                    m21log("iter", iter);
                    m21log("chisq", chisq);
                }
                if (done == NDONE) alamda = 0.;
                for (j = 1; j <= mfit; j++) {
                    for (k = 1; k <= mfit; k++) covar(j, k) = alpha(j, k);
                    covar(j, j) = alpha(j, j) * (1.0 + alamda);
                    for (k = 1; k <= mfit; k++) temp(j, k) = covar(j, k);
                    oneda(j, 1) = beta(j);
                }
                GaussJordanElimination gje;
                if (!gje.solve(temp, oneda)) {
                    temp.log("temp");
                    oneda.log("oneda");
                    alpha.log("alpha");
                    beta.log("beta");
                    return 0;
                }
//                try {
//                } catch (const math21::fatal_error &e) {
//                    MATH21_ASSERT(0)
//                }
                // get covariance matrix which is inverse of alpha.
                for (j = 1; j <= mfit; j++) {
                    for (k = 1; k <= mfit; k++) covar(j, k) = temp(j, k);
                    da(j) = oneda(j, 1);
                }
                if (done == NDONE) {
                    covsrt(covar);
                    covsrt(alpha);
                    if (isLog) {
                        m21log("iter", iter);
                        m21log("chisq", chisq);
                    }
                    return 1;
                }
                for (j = 1, l = 1; l <= ma; l++)
                    if (ia(l)) atry(l) = a(l) + da(j++);
                // covar, da used as temporary space for alpha and beta respectively.
                mrqcof(atry, covar, da);
                if (xjabs(chisq - ochisq) < xjmax(tol, tol * chisq)) done++;
                if (chisq < ochisq) {
                    alamda *= 0.1;
                    ochisq = chisq;
                    for (j = 1; j <= mfit; j++) {
                        for (k = 1; k <= mfit; k++) alpha(j, k) = covar(j, k);
                        beta(j) = da(j);
                    }
                    for (l = 1; l <= ma; l++) a(l) = atry(l);
                } else {
                    alamda *= 10.0;
                    chisq = ochisq;
                }
            }
            m21warn("Fitmrq too many iterations");
            return 0;
        }

        void Fitmrq::mrqcof(const VecR &a, MatR &alpha, VecR &beta) {
            NumN i, j, k, l, m;
            NumR ymod, wt, sig2i, dy;
            VecR dyda(ma);
            for (j = 1; j <= mfit; j++) {
                for (k = 1; k <= j; k++) alpha(j, k) = 0.0;
                beta(j) = 0.;
            }
            chisq = 0.;
            for (i = 1; i <= ndat; i++) {
                funcs(x(i), a, ymod, dyda);
                sig2i = 1.0 / (sig(i) * sig(i));
                dy = y(i) - ymod;
                for (j = 1, l = 1; l <= ma; l++) {
                    if (ia(l)) {
                        wt = dyda(l) * sig2i;
                        for (k = 1, m = 1; m <= l; m++)
                            if (ia(m)) alpha(j, k++) += wt * dyda(m);
                        beta(j++) += dy * wt;
                    }
                }
                chisq += dy * dy * sig2i;
            }
            for (j = 2; j <= mfit; j++)
                for (k = 1; k < j; k++) alpha(k, j) = alpha(j, k);
        }

        void Fitmrq::covsrt(MatR &covar) {
            NumN i, j, k;
            for (i = mfit + 1; i <= ma; i++)
                for (j = 1; j <= i; j++) covar(i, j) = covar(j, i) = 0.0;
            k = mfit;
            for (j = ma; j >= 1; j--) {
                if (ia(j)) {
                    for (i = 1; i <= ma; i++) m21_swap(covar(i, k), covar(i, j));
                    for (i = 1; i <= ma; i++) m21_swap(covar(k, i), covar(j, i));
                    k--;
                }
            }
        }

        void fgauss(const NumR x, const VecR &a, NumR &y, VecR &dyda) {
            NumN i, na = a.size();
            NumR fac, ex, arg;
            y = 0.;
            for (i = 1; i <= na; i += 3) {
                arg = (x - a(i + 1)) / a(i + 2);
                ex = xjexp(-xjsquare(arg));
                fac = a(i) * ex * 2. * arg;
                y += a(i) * ex;
                dyda(i) = ex;
                dyda(i + 1) = fac / a(i + 2);
                dyda(i + 2) = fac * arg / a(i + 2);
            }
        }
    }

    void math21_sample_1d(const VecR &x, VecR &y, const VecR &a, void funks(const NumR, const VecR &, NumR &, VecR &)) {
        MATH21_ASSERT(y.size() == x.size())
        NumN ndat = x.size();
        VecR dyda(a.size());
        for (NumN i = 1; i <= ndat; ++i) {
            funks(x(i), a, y.at(i), dyda);
        }
    }

    void math21_numerical_recipes_Fitmrq_test() {
        using namespace numerical_recipes;

        NumN na = 6;

        VecR a0(na);
        VecR a(na);
        a = 3, 5, 2, 4, 3, 5;

        NumN ndat = 1000;
        VecR x(ndat);
        VecR y(ndat);
        VecR ssig(ndat);


        RanUniform ran;
        ran.set(-7, 7);
        math21_random_draw(x, ran);
        ssig = 1;
        math21_sample_1d(x, y, a, fgauss);
        x.log("x");
        y.log("y");
        a.log("a");

        VecR err(na);
        RanNormal ranNormal;
        ranNormal.set(0, 1.2);
        math21_random_draw(err, ranNormal);
        math21_operator_add(a, err, a0);

//        math21_random_draw(a0, ran);
        a0.log("a0");
        Fitmrq fitmrq(x, y, ssig, a0, fgauss);
        fitmrq.fit(1);
        fitmrq.getOptimal().log("optimal");
    }
}