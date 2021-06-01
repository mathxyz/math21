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

#include "svd.h"
#include "eigen_sym.h"
#include "ops.h"

namespace math21 {
    using namespace numerical_recipes;

    // see http://fourier.eng.hmc.edu/e176/lectures/NM/node8.html
    // A = U*W*V.t
    void math21_operator_svd_real(const MatR &M, MatR &u, VecR &w, MatR &v) {
        ShiftedMatR A;
        A.setTensor(M);
        SVD svd(A);
        u.copyFrom(svd.get_u().getTensor());
        w.copyFrom(svd.get_w().getTensor());
        v.copyFrom(svd.get_v().getTensor());
    }

    void math21_operator_eigen_real_sys_descending(const MatR &M, VecR &Lambda, MatR &X) {
        ShiftedMatR A;
        A.setTensor(M);
        Symmeig module(A);
//        Jacobi module(A);
        Lambda.copyFrom(module.get_eigenvalues().getTensor());
        X.copyFrom(module.get_eigenvectors().getTensor());
    }

    void math21_operator_eigen_real_sys_ascending(const MatR &M, VecR &Lambda, MatR &X) {
        ShiftedMatR A;
        A.setTensor(M);
        Symmeig module(A);
//        Jacobi module(A);
        Lambda.copyFrom(module.get_eigenvalues().getTensor());
        X.copyFrom(module.get_eigenvectors().getTensor());
        math21_operator_matrix_reverse_x_axis(Lambda);
        math21_operator_matrix_reverse_y_axis(X);
    }
}