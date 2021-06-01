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

        // Computes all eigenvalues and eigenvectors of a real symmetric matrix by Jacobi’s method.
        // Computes all eigenvalues and eigenvectors of a real symmetric matrix a[0..n-1][0..n-1].
        //On output, d[0..n-1] contains the eigenvalues of a sorted into descending order, while
        //v[0..n-1][0..n-1] is a matrix whose columns contain the corresponding normalized eigenvectors.
        //nrot contains the number of Jacobi rotations that were required. Only the upper
        //triangle of a is accessed.
        struct Jacobi {
            NumZ n;
            ShiftedMatR a, v;
            ShiftedVecR d;
            NumZ nrot;
            NumR EPS;

            Jacobi(const ShiftedMatR &aa);

            inline void rot(ShiftedMatR &a, NumR s, NumR tau, NumZ i,
                            NumZ j, NumZ k, NumZ l);

            // in descending order
            const ShiftedVecR &get_eigenvalues() const {
                return d;
            }

            // the columns contain the corresponding normalized eigenvectors
            const ShiftedVecR &get_eigenvectors() const {
                return v;
            }
        };

        // Computes all eigenvalues and eigenvectors of a real symmetric matrix by reduction to tridiagonal
        // form followed by QL iteration.
        struct Symmeig {
        private:
            NumZ n;
            ShiftedMatR z;
            ShiftedVecR d, e;
            NumB yesvecs;

            void sort();

            void tred2();

            void tqli();

            static NumR pythag(NumR a, NumR b);

        public:
            // you can suppress the computation of eigenvectors by setting the default argument
            // to false.
            Symmeig(const ShiftedMatR &a, NumB yesvec = true);

            // If you already have a matrix in tridiagonal form, you use the constructor, which
            // accepts the diagonal and subdiagonal of the matrix as vectors.
            Symmeig(const ShiftedVecR &dd, const ShiftedVecR &ee, NumB yesvec = true);

            // in descending order
            const ShiftedVecR &get_eigenvalues() const {
                return d;
            }

            // the columns contain the corresponding normalized eigenvectors
            const ShiftedVecR &get_eigenvectors() const {
                return z;
            }
        };
    }
}