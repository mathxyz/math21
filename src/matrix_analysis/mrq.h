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

namespace math21 {
    namespace numerical_recipes {
        struct Fitmrq {
        private:
            static const NumN NDONE = 4, ITMAX = 1000;
            NumN ndat, ma, mfit;
            const VecR &x, &y, &sig;
            NumR tol; // tolerance

            void (*funcs)(const NumR, const VecR &, NumR &, VecR &);

            VecB ia;
            VecR a;
            MatR covar;
            MatR alpha;
            NumR chisq;

            void mrqcof(const VecR &a, MatR &alpha, VecR &beta);

            void covsrt(MatR &covar);

        public:
            const VecR &getOptimal() const {
                return a;
            }

            // data not set zero
            Fitmrq(const VecR &xx, const VecR &yy, const VecR &ssig, const VecR &aa,
                   void funks(const NumR, const VecR &, NumR &, VecR &),
                   const NumR TOL = 1.e-3) : ndat(xx.size()), ma(aa.size()), x(xx), y(yy), sig(ssig),
                                             tol(TOL), funcs(funks), ia(ma), alpha(ma, ma), a(aa), covar(ma, ma) {
                for (NumN i = 1; i <= ma; i++) ia(i) = XJ_TRUE;
            }

            void hold(const NumN i, const NumR val) {
                ia(i) = XJ_FALSE;
                a(i) = val;
            }

            void free(const NumN i) { ia(i) = XJ_TRUE; }

            NumB fit(NumB isLog=0);
        };

        void fgauss(const NumR x, const VecR &a, NumR &y, VecR &dyda);
    }

    void math21_numerical_recipes_Fitmrq_test();
}