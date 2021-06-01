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

namespace math21 {
    namespace numerical_recipes {
        //A*(X,Y)=(B,I), A will become Y, B will become X.
        NumB GaussJordanElimination::solve(MatR &A, MatR &B) {
            MATH21_ASSERT(A.nrows() == A.ncols());
            NumN i, icol, irow, j, k, l, ll, n = A.nrows(), m = B.ncols();
            NumR big, dum, pivinv;
            ArrayN index_c(n), index_r(n), ipiv(n);
            for (j = 1; j <= n; j++) ipiv(j) = 0;
            for (i = 1; i <= n; i++) {
                big = 0.0;
                for (j = 1; j <= n; j++)
                    if (ipiv(j) != 1)
                        for (k = 1; k <= n; k++) {
                            if (ipiv(k) == 0) {
                                if (xjabs(A(j, k)) >= big) {
                                    big = xjabs(A(j, k));
                                    irow = j;
                                    icol = k;
                                }
                            }
                        }
                ++(ipiv(icol));
                if (irow != icol) {
                    for (l = 1; l <= n; l++) m21_swap(A(irow, l), A(icol, l));
                    for (l = 1; l <= m; l++) m21_swap(B(irow, l), B(icol, l));
                }
                index_r(i) = irow;
                index_c(i) = icol;
                if (A(icol, icol) == 0.0) {
                    m21warn("Gauss-Jordan elimination: Singular Matrix");
                    return 0;
                }
                pivinv = 1.0 / A(icol, icol);
                A(icol, icol) = 1.0;
                for (l = 1; l <= n; l++) A(icol, l) *= pivinv;
                for (l = 1; l <= m; l++) B(icol, l) *= pivinv;
                for (ll = 1; ll <= n; ll++)
                    if (ll != icol) {
                        dum = A(ll, icol);
                        A(ll, icol) = 0.0;
                        for (l = 1; l <= n; l++) A(ll, l) -= A(icol, l) * dum;
                        for (l = 1; l <= m; l++) B(ll, l) -= B(icol, l) * dum;
                    }
            }
            for (l = n; l >= 1; l--) {
                if (index_r(l) != index_c(l))
                    for (k = 1; k <= n; k++)
                        m21_swap(A(k, index_r(l)), A(k, index_c(l)));
            }
            return 1;
        }

        //A*Y=I, A will become Y.
        NumB GaussJordanElimination::solve(MatR &A) {
            MatR B;
            B.setSize(A.nrows(), 0); // B will be empty.
            return solve(A, B);
        }
    }
}