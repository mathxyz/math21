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
        // Book: numerical recipes (the art of scientific computing), 3rd edition

        // todo: test
        // A = U*W*V.t
        // Any M*N matrix A can be written as the product of an M*N
        // column-orthogonal matrix U, an N*N diagonal matrix W with positive or zero
        // elements (the singular values), and the transpose of an N*N orthogonal matrix V.
        struct SVD : public think::Algorithm {
        private:
            NumN m, n;
            ShiftedMatR u, v; // The matrices U and V
            ShiftedVecR w; // The diagonal matrix W
            NumR eps, tsh;

            void decompose();

            void reorder();

            static NumR pythag(NumR a, NumR b);

        public:

            SVD(const ShiftedMatR &a);

            // Solve A*x=b for a vector x using the pseudoinverse of A as obtained by SVD.
            void solve_vec(const ShiftedVecR &b, ShiftedVecR &x, NumR thresh = -1.);

            // Solves m sets of n equations A*X=B using the pseudoinverse of A.
            void solve_mat(const ShiftedMatR &b, ShiftedMatR &x, NumR thresh = -1.);

            NumN rank(NumR thresh = -1.);

            NumN nullity(NumR thresh = -1.);

            // Give an orthonormal basis for the range of A as the columns of a returned matrix.
            void range(ShiftedMatR &rnge, NumR thresh = -1.);

            // Give an orthonormal basis for the nullspace of A as the columns of a returned matrix.
            void nullspace(ShiftedMatR &nullsp, NumR thresh = -1.);

            // return reciprocal of the condition number of A
            NumR inv_condition();

            const ShiftedMatR &get_u() const {
                return u;
            }

            const ShiftedMatR &get_v() const {
                return v;
            }

            const ShiftedVecR &get_w() const {
                return w;
            }
        };
    }
}
