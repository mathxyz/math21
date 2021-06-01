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

#include "01.h"
#include "02.h"
#include "differential_operators.h"
#include "test_01_c.h"
#include "../../point/files.h"

using namespace math21;
using namespace ad;

// f(x) = sum f^(n)(x0) * (x-x0)^n/n!
// let x0 = 0, then
// sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...+ (-1)^n*(x^(2n+1))/(2n+1)! + ...
// Taylor approximation to sin function
ad_point ad_sin_taylor_appr(const ad_point &x) {
    auto currterm = x;
    auto ans = currterm;
    for (NumN n = 1; n <= 1000; ++n) {
        m21logNoNewLine(n);
        currterm = -currterm * ad_power(x, 1, 2) / ((2 * n) * (2 * n + 1));
        ans = ans + currterm;
        if (xjabs(ad_get_variable(currterm).getValue()(1)) < 0.2) { break; }
        m21logNoNewLine(" ");
    }
    m21logNoNewLine("\n");
    return ans;
}

NumR math21_test_c_ad_sin_dn(NumR x0, NumN order) {
    ad_clear_graph();
    TenR x_value(1);
    x_value = x0;
    auto x = ad_create_point_input_with_value(x_value);
    auto y = ad_sin(x);
    for (NumN i = 1; i <= order; ++i) {
        y = grad(x, y);
    }
    return ad_get_variable(y).getValue()(1);
}

NumR math21_test_c_ad_sin_taylor_appr_dn(NumR x0, NumN order) {
    ad_clear_graph();
    TenR x_value(1);
    x_value = x0;
    auto x = ad_create_point_input_with_value(x_value);
    auto y = ad_sin_taylor_appr(x);
    for (NumN i = 1; i <= order; ++i) {
        y = grad(x, y);
    }
    return ad_get_variable(y).getValue()(1);
}

m21point math21_test_ad_tanh_like_dn(m21point px, NumN order) {
    NumR k = 0.5;
    ad_clear_graph();
    m21point y_empty = {0};
    if (math21_type_get<TenR>() != px.type) {
        m21warn("cast error");
        return y_empty;
    }
    TenR &x_value = math21_cast_to_T<TenR>(px);
    auto x = ad_create_point_input_with_value(x_value);
    auto y = tanh(k * x);
    for (NumN i = 1; i <= order; ++i) {
//        m21log("size", ad_get_data().size());
        y = egrad(x, y);
        if (y.id == 0) {
            m21log("f=0, so stop computing f'");
            m21log("i", i);
            break;
        }
    }
    return math21_cast_to_point(ad_get_variable(y).getValue());
}

m21point math21_test_ad_logsumexp_like(m21point px) {
    ad_clear_graph();
    m21point y_empty = {0};
    if (math21_type_get<TenR>() != px.type) {
        m21warn("cast error");
        return y_empty;
    }
    TenR &x_value = math21_cast_to_T<TenR>(px);
    auto x = ad_create_point_input_with_value(x_value);
    auto y = x - ad_logsumexp(x);
    return math21_cast_to_point(ad_get_variable(y).getValue());
}

m21point math21_test_ad_gmm_log_likelihood_dn(m21point params0, m21point data0,
                                              NumN n_component, NumN n_feature, NumN order) {
    ad_clear_graph();
    m21point y_empty = {0};
    if (math21_type_get<TenR>() != params0.type || math21_type_get<TenR>() != data0.type) {
        m21warn("cast error");
        return y_empty;
    }
    TenR &params_value = math21_cast_to_T<TenR>(params0);
    ad_point params(params_value, 1);
    TenR &data = math21_cast_to_T<TenR>(data0);
    auto y = ad_gmm_log_likelihood(params, data,
                                   n_component, n_feature, 0);
    for (NumN i = 1; i <= order; ++i) {
        y = grad(params, y);
        if (y.id == 0) {
            m21log("f=0, so stop computing f'");
            m21log("i", i);
            break;
        }
    }
    return math21_cast_to_point(ad_get_variable(y).getValue());
}

m21point math21_test_ad_get_f_gmm_log_likelihood(m21point params0, m21point data0,
                                                 NumN n_component, NumN n_feature) {
    return math21_test_ad_get_f_gmm_log_likelihood_with_order(params0, data0, n_component, n_feature, 0);
}

m21point math21_test_ad_get_f_gmm_log_likelihood_with_order(m21point params0, m21point data0,
                                                            NumN n_component, NumN n_feature, NumB isECorder) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != params0.type || math21_type_get<TenR>() != data0.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto &params = math21_cast_to_T<ad_point>(params0);
    TenR &data = math21_cast_to_T<TenR>(data0);
    auto py = ad_gmm_log_likelihood(params, data,
                                    n_component, n_feature, isECorder);
    auto y = new ad_point();
    *y = py;
    return math21_cast_to_point(*y);
}

m21point math21_point_ad_grad(m21point x, m21point y) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != x.type || math21_type_get<ad_point>() != y.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto &px = math21_cast_to_T<ad_point>(x);
    auto &py = math21_cast_to_T<ad_point>(y);
    auto *dy = new ad_point();
    *dy = grad(px, py);
    if (dy->id == 0) {
        m21warn("id=0");
    }
    return math21_cast_to_point(*dy);
}

m21point math21_point_ad_hessian_vector_product(m21point x, m21point y, m21point v) {
    m21point y_empty = {0};
    if (math21_type_get<ad_point>() != x.type || math21_type_get<ad_point>() != y.type
        || math21_type_get<ad_point>() != v.type) {
        m21warn("cast error");
        return y_empty;
    }
    auto &px = math21_cast_to_T<ad_point>(x);
    auto &py = math21_cast_to_T<ad_point>(y);
    auto &pv = math21_cast_to_T<ad_point>(v);
    auto *dy = new ad_point();
    *dy = ad_hessian_vector_product(px, py, pv);
    if (dy->id == 0) {
        m21warn("id=0");
    }
    return math21_cast_to_point(*dy);
}

void math21_point_ad_fv(m21point x) {
    ad_fv(math21_cast_to_T<ad_point>(x));
}

void math21_ad_clear_graph() {
    ad_clear_graph();
}

