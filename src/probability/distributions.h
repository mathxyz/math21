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
#include "ran.h"

namespace math21 {

    void math21_random_draw(NumN &x, think::Random &ran);

    void math21_random_draw(NumZ &x, think::Random &ran);

    void math21_random_draw(NumR &x, think::Random &ran);


    template<typename T, template<typename> class Container>
    void math21_random_container_draw(Container<T> &A, think::Random &ran) {
        MATH21_ASSERT(!A.isEmpty());
        NumN i;
        NumN n = A.size();
//#pragma omp parallel for
        for (i = 1; i <= n; ++i) {
            math21_random_draw(A.at(i), ran);
        }
    }

    template<typename T>
    void math21_random_draw(Tensor <T> &m, think::Random &ran) {
        MATH21_ASSERT(!m.isEmpty());
        return math21_random_container_draw(m, ran);
    }

    template<typename T>
    void math21_op_random_draw(Tensor <T> &m, think::Random &ran) {
        MATH21_ASSERT(!m.isEmpty());
        if (m.is_cpu()) {
            math21_random_container_draw(m, ran);
        } else {
            Tensor<T> m_c;
            m_c.setSize(m.shape());
            math21_random_container_draw(m_c, ran);
            m = m_c;
        }
    }

    NumR math21_pr_poisson_probability(NumN n, NumN lambda);

    NumR math21_pr_binomial(NumN n, NumR p, NumN k);

    NumB math21_pr_mvn_logpdf2(const MatR &x, const VecR &mean, const MatR &covariance, VecR &value);

    void math21_pr_mvn_dYdX_diag_logpdf(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dx);

    void math21_pr_mvn_dYdmu_logpdf(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dmu);

    void math21_pr_mvn_dYdSig_logpdf(const VecR &x, const VecR &mean, const MatR &covariance, MatR &dSig);
}