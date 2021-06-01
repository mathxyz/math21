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

#include "../functions_01/files.h"
#include "../functions_02/files.h"
#include "../functions_03/files.h"
#include "../functions_04/files.h"
#include "../differential.h"
#include "point.h"
#include "01.h"

namespace math21 {
    namespace ad {

        NumN ad_get_max_device_type(const ad_point &x1, const ad_point &x2) {
            return xjmax(ad_get_device_type(x1), ad_get_device_type(x2));
        }

        NumN ad_get_max_device_type(const ad_point &x1, const ad_point &x2, const ad_point &x3) {
            return xjmax(xjmax(ad_get_device_type(x1), ad_get_device_type(x2)), ad_get_device_type(x3));
        }

        ad_point ad_to_device_type(const ad_point &x, NumN deviceType) {
            if (ad_get_device_type(x) != deviceType) {
                if (ad_point_is_cpu(x)) {
                    return ad_push(x);
                } else {
                    return ad_pull(x);
                }
            } else {
                return x;
            }
        }

        void ad_to_same_device(const ad_point &x1, const ad_point &x2, ad_point &y1, ad_point &y2) {
            NumN deviceType = ad_get_max_device_type(x1, x2);
            y1 = ad_to_device_type(x1, deviceType);
            y2 = ad_to_device_type(x2, deviceType);
        }

        void ad_to_same_device(const ad_point &x1, const ad_point &x2, const ad_point &x3,
                               ad_point &y1, ad_point &y2, ad_point &y3) {
            NumN deviceType = ad_get_max_device_type(x1, x2);
            y1 = ad_to_device_type(x1, deviceType);
            y2 = ad_to_device_type(x2, deviceType);
            y3 = ad_to_device_type(x3, deviceType);
        }

        ad_point ad_sin(const ad_point &p) {
            op_sin sin0;
            Function &function = sin0;
            NumN x = p.id;
            NumN y = function.evaluate(x, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_cos(const ad_point &p) {
            op_cos cos0;
            Function &function = cos0;
            NumN x = p.id;
            NumN y = function.evaluate(x, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_sum(const ad_point &p) {
            op_sum sum;
            Function &function = sum;
            NumN x = p.id;
            NumN y = function.evaluate(x, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_sum(const ad_point &p, const ad_point &axes, NumB isKeepingDims) {
            op_sum sum(isKeepingDims);
            Function &function = sum;
            NumN y = function.evaluate(p.id, axes.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_tensor_broadcast(const ad_point &p, const ad_point &d) {
            op_broadcast_tensor sum;
            Function &function = sum;
            NumN y = function.evaluate(p.id, d.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_add(const ad_point &_p1, const ad_point &_p2) {
            ad_point p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            op_add add;
            Function &function = add;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN y = function.evaluate(x1, x2, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_add(const ad_point &_p1, const ad_point &_p2, const ad_point &_p3) {
            ad_point p1, p2, p3;
            ad_to_same_device(_p1, _p2, _p3, p1, p2, p3);
            op_add add;
            Function &function = add;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN x3 = p3.id;
            NumN y = function.evaluate(x1, x2, x3, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_negate(const ad_point &p) {
            TenR m1;
            m1.setSize(1);
            m1 = -1;
            ad_point pm1(m1, 0, ad_get_device_type(p));
            return ad_mul(pm1, p);
        }

        ad_point ad_subtract(const ad_point &_p1, const ad_point &_p2) {
            ad_point p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            return ad_add(p1, ad_negate(p2));
        }

        ad_point ad_mul(const ad_point &_p1, const ad_point &_p2) {
            ad_point p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            op_multiply mul;
            Function &function = mul;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN y = function.evaluate(x1, x2, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mul(const ad_point &_p1, const ad_point &_p2, const ad_point &_p3) {
            ad_point p1, p2, p3;
            ad_to_same_device(_p1, _p2, _p3, p1, p2, p3);
            op_multiply mul;
            Function &function = mul;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN x3 = p3.id;
            NumN y = function.evaluate(x1, x2, x3, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_divide(const ad_point &_p1, const ad_point &_p2) {
            ad_point p1, p2;
            ad_to_same_device(_p1, _p2, p1, p2);
            return ad_mul(p1, ad_power(p2, 1, -1));
        }

        // Device type of y is based on that of x.
        // k is a number on cpu or not.
        // y = kx
        ad_point ad_kx(const ad_point &k, const ad_point &x) {
            op_kx _;
            Function &f = _;
            NumN y = f.evaluate(k.id, x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point operator+(const ad_point &p1, const ad_point &p2) {
            return ad_add(p1, p2);
        }

        ad_point operator-(const ad_point &p1) {
            return ad_negate(p1);
        }

        ad_point operator-(const ad_point &p1, const ad_point &p2) {
            return ad_subtract(p1, p2);
        }

        ad_point operator*(const ad_point &p1, const ad_point &p2) {
            return ad_mul(p1, p2);
        }

        ad_point operator/(const ad_point &p1, const ad_point &p2) {
            return ad_divide(p1, p2);
        }

        ad_point ad_power(const ad_point &x, NumR k, NumR p) {
            op_power power(k, p);
            Function &function = power;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_exp(const ad_point &x) {
            op_exp exp;
            Function &function = exp;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point exp(const ad_point &x) {
            return ad_exp(x);
        }

        ad_point ad_exp(const ad_point &x, NumR base) {
            MATH21_ASSERT(0)
            return ad_point(0, 0);
        }

        ad_point ad_log(const ad_point &x) {
            op_log log;
            Function &function = log;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_log(const ad_point &x, NumR base) {
            op_log log(base);
            Function &function = log;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mat_trans(const ad_point &x) {
            op_mat_trans _ad_mat_trans;
            Function &function = _ad_mat_trans;
            NumN y = function.evaluate(x.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mat_mul(const ad_point &p1, const ad_point &p2) {
            op_mat_mul mul0;
            Function &function = mul0;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN y = function.evaluate(x1, x2, ad_global_get_data());
//            function.forward(x1, x2, y, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_inner_product(const ad_point &p1, const ad_point &p2) {
            op_inner_product inner_product0;
            Function &function = inner_product0;
            NumN x1 = p1.id;
            NumN x2 = p2.id;
            NumN y = function.evaluate(x1, x2, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point dot(const ad_point &p1, const ad_point &p2) {
            return ad_inner_product(p1, p2);
        }

        ad_point ad_at(const ad_point &p, NumN index) {
            VecR b;
            auto &d = ad_get_variable(p).getValue().shape();
            b.setSize(d);
            b.zeros();
            b(index) = 1;
            return ad_inner_product(p, b);
        }

        ad_point at(const ad_point &p, NumN index) {
            return ad_at(p, index);
        }

        ad_point ad_push(const ad_point &p) {
            op_push _op_push;
            Function &f_op_push = _op_push;
            NumN x = p.id;
            NumN y = f_op_push.evaluate(x, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_pull(const ad_point &p) {
            op_pull _op_pull;
            Function &f_op_pull = _op_pull;
            NumN x = p.id;
            NumN y = f_op_pull.evaluate(x, ad_global_get_data());
            return ad_point(y, 0);
        }

        // value is a number and d is shape.
        ad_point ad_create_using_shape(const ad_point &value, const ad_point &d) {
            op_create _;
            Function &f = _;
            NumN y = f.evaluate(value.id, d.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mean(const ad_point &p) {
            op_mean _;
            Function &f = _;
            NumN y = f.evaluate(p.id, ad_global_get_data());
            return ad_point(y, 0);
        }

        ad_point ad_mean(const ad_point &p, const ad_point &axes, const ad_point &isKeepingDims) {
            op_mean _;
            Function &f = _;
            NumN y = f.evaluate(p.id, axes.id, isKeepingDims.id, ad_global_get_data());
            return ad_point(y, 0);
        }

    }
}