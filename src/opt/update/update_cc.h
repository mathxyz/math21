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

#include "inner_cc.h"

namespace math21 {
    template<typename T>
    void math21_op_optimization_adam_update_part_2(
            Tensor <T> &x, const Tensor <T> &m, const Tensor <T> &v,
            NumR beta1, NumR beta2, NumR alpha, NumR eps, NumN t) {
        if (x.is_cpu()) {
            math21_generic_optimization_adam_update_part_2_cpu(
                    x.size(), x.getDataAddress(), m.getDataAddress(), v.getDataAddress(),
                    beta1, beta2, alpha, eps, t, x.getSpace().type);
        } else {
            math21_generic_optimization_adam_update_part_2_wrapper(
                    x.size(), x.getDataAddressWrapper(), m.getDataAddressWrapper(), v.getDataAddressWrapper(),
                    beta1, beta2, alpha, eps, t, x.getSpace().type);
        }
    }

    template<typename T>
    void math21_op_optimization_adam_update(Tensor <T> &x, Tensor <T> &neg_dx, Tensor <T> &m,
                                        Tensor <T> &v, NumR beta1, NumR beta2,
                                        NumR eps, NumR decay, NumR alpha, NumN mini_batch_size,
                                        NumN t) {
        // dL/dx = dL/dx + decay * x
        if (decay != 0) {
            math21_op_vector_kx_add_y(-decay * mini_batch_size, x, neg_dx);
        }

        // update biased first moment estimate
        // m = beta1 * m + (1-beta1)*dL/dx
        // m = beta1 * m
        math21_op_vector_kx_onto(beta1, m);
        // m = m + (1-beta1)*dL/dx
        math21_op_vector_kx_add_y((1 - beta1), neg_dx, m);

        // update biased second raw moment estimate
        // v = beta2 * v + (1-beta2)*(dL/dx * dL/dx)
        // v = beta2 * v
        math21_op_vector_kx_onto(beta2, v);
        // dL/dx <- dL/dx * dL/dx
        math21_op_vector_xy(neg_dx, neg_dx);
        // v = v + (1-beta2)*(dL/dx * dL/dx)
        math21_op_vector_kx_add_y((1 - beta2), neg_dx, v);

        math21_op_optimization_adam_update_part_2(x, m, v, beta1, beta2, alpha, eps, t);
        // dL/dx <- 0
        math21_op_vector_set_value(neg_dx, 0);
    }
}