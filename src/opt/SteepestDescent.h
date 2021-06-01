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

    struct sd_update_rule {
    public:
        Functional &f;
        VecR x;
        NumR y_old, y;
        NumN time, time_max;

        sd_update_rule(Functional &f) : f(f) {
            time = 0;
            time_max = 1000;
        }

        virtual void update() = 0;

        void setInit(const VecR &_x) {
            MATH21_ASSERT(!x.isEmpty())
            x.assign(_x);
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_normal : public sd_update_rule {
    private:
        NumR epsilon, epsilon_0, epsilon_tao;
    public:
        NumN tao;

        sd_update_rule_normal(Functional &f) : sd_update_rule(f) {
            epsilon_0 = 0.1;
            epsilon_tao = 0.01 * epsilon_0;
            tao = 10000;
            x.setSize(f.getXDim());
            y = f.valueAt(x);
        }

        void update() {
            if (time < tao) {
                NumR alpha = time / (NumR) tao;
                epsilon = (1 - alpha) * epsilon_0 + alpha * epsilon_tao;
            } else {
                epsilon = epsilon_tao;
            }
//        x.log("x");
            const VecR &g = f.derivativeValueAt(x);
//        g.log("g");
            math21_operator_linear_to_A(1, x, -epsilon, g);
//        x.log("x");
            y_old = y;
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_momentum : public sd_update_rule {
    private:
        NumR alpha, epsilon;
        VecR v;
    public:

        sd_update_rule_momentum(Functional &f) : sd_update_rule(f) {
            alpha = 0.9;
            epsilon = 0.1;
            x.setSize(f.getXDim());
            v.setSize(f.getXDim());
            x = 0;
            v = 0;
            y = f.valueAt(x);
        }

        void update() {
            const VecR &g = f.derivativeValueAt(x);
            math21_operator_linear_to_A(alpha, v, -epsilon, g);
            math21_operator_addToA(x, v);
            y_old = y;
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_Nesterove_momentum : public sd_update_rule {
    private:
        NumR alpha, epsilon;
        VecR v;
    public:

        sd_update_rule_Nesterove_momentum(Functional &f) : sd_update_rule(f) {
            alpha = 0.9;
            epsilon = 0.1;
            x.setSize(f.getXDim());
            v.setSize(f.getXDim());
            x = 0;
            v = 0;
            y = f.valueAt(x);
        }

        void update() {
            VecR x_hat;
            math21_operator_linear(1, x, alpha, v, x_hat);
            const VecR &g = f.derivativeValueAt(x_hat);
            math21_operator_linear_to_A(alpha, v, -epsilon, g);
            math21_operator_addToA(x, v);
            y_old = y;
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_AdaGrad : public sd_update_rule {
    private:
        NumR delta;
        NumR epsilon;
        VecR r;
        VecR delta_x;

        VecR tmp_1;
    public:

        sd_update_rule_AdaGrad(Functional &f) : sd_update_rule(f) {
            delta = MATH21_10NEG7;
            epsilon = 0.1;
            x.setSize(f.getXDim());
            r.setSize(f.getXDim());
            tmp_1.setSize(f.getXDim());
            delta_x.setSize(f.getXDim());
            x = 0;
            r = 0;
            y = f.valueAt(x);
        }

        void update() {
            const VecR &g = f.derivativeValueAt(x);
            math21_operator_SchurProduct(g, g, tmp_1);
            math21_operator_addToA(r, tmp_1);
//            VecR r_tmp;
//            r_tmp = m21sqrt(m21add(delta, r));
//            MATH21_ASSERT(argmin(r_tmp)>0, "r_tmp should be larger than zero!");
//            delta_x = m21divide((-epsilon), r_tmp).SchurProduct(g);
            VecR &tmp_2 = tmp_1;
            math21_operator_add(delta, r, tmp_2);
            math21_operator_sqrt_to(tmp_2);
            math21_operator_divide_to(-epsilon, tmp_2);
            math21_operator_SchurProduct(tmp_2, g, delta_x);
//            delta_x = m21divide((-epsilon), m21sqrt(m21add(delta, r))).SchurProduct(g);
            math21_operator_addToA(x, delta_x);
            y_old = y;
            y = f.valueAt(x);
        }
    };


    struct sd_update_rule_RMSProp : public sd_update_rule {
    private:
        NumR delta;
        NumR rho;
        NumR epsilon;
        VecR r;
        VecR tmp_1;
    public:

        sd_update_rule_RMSProp(Functional &f) : sd_update_rule(f) {
            rho = 0.9;
            delta = MATH21_10NEG7;
            epsilon = 0.01;
            x.setSize(f.getXDim());
            r.setSize(f.getXDim());
            tmp_1.setSize(f.getXDim());
            x = 0;
            r = 0;
            y = f.valueAt(x);
        }

        void update() {
            const VecR &g = f.derivativeValueAt(x);
            math21_operator_SchurProduct(g, g, tmp_1);
            math21_operator_linear_to_A(rho, r, (1 - rho), tmp_1);

            VecR &tmp_2 = tmp_1;
            math21_operator_add(delta, r, tmp_2);
            math21_operator_sqrt_to(tmp_2);
            math21_operator_divide_to(-epsilon, tmp_2);

            VecR &delta_x = tmp_2;
            math21_operator_SchurProduct_to_A(delta_x, g);

            math21_operator_addToA(x, delta_x);
            y_old = y;
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_RMSProp_Nesterov_momentum : public sd_update_rule {
    private:
        NumR delta;
        NumR rho;
        NumR epsilon;
        VecR r;
        NumR alpha;
        VecR v;

        //local
        VecR tmp;
    public:

        sd_update_rule_RMSProp_Nesterov_momentum(Functional &f) : sd_update_rule(f) {
            alpha = 0.9;
            v.setSize(f.getXDim());
            rho = 0.9;
            delta = MATH21_10NEG7;
            epsilon = 0.01;
            x.setSize(f.getXDim());
            tmp.setSize(f.getXDim());
            r.setSize(f.getXDim());
            x = 0;
            r = 0;
            DefaultRandomEngine engine(21);
            RanUniform ranUniform(engine);
            ranUniform.set(-0.8, 0.8);
            math21_random_draw(x, ranUniform);
            y = f.valueAt(x);
        }

        void update() {
            VecR &x_hat = tmp;
            math21_operator_linear(1, x, alpha, v, x_hat);
            const VecR &g = f.derivativeValueAt(x_hat);
            VecR &g_2 = tmp;
            math21_operator_SchurProduct(g, g, g_2);
            math21_operator_linear_to_A(rho, r, (1 - rho), g_2);

            // invalidate x_hat
            VecR &tmp_2 = tmp;
            math21_operator_add(delta, r, tmp_2);
            math21_operator_sqrt_to(tmp_2);
            math21_operator_divide_to(-epsilon, tmp_2);
            math21_operator_SchurProduct_to_A(tmp_2, g);
            math21_operator_linear_to_A(alpha, v, 1, tmp_2);
            math21_operator_addToA(x, v);
            y_old = y;
            y = f.valueAt(x);
        }
    };

    struct sd_update_rule_Adam : public sd_update_rule {
    private:
        NumR delta;
        NumR rho1, rho2;
        NumR rho1_product, rho2_product;
        NumR epsilon;
        VecR s, r;
        VecR s_hat, r_hat;

        VecR tmp;
    public:
        sd_update_rule_Adam(Functional &f) : sd_update_rule(f) {
            s.setSize(f.getXDim());
            r.setSize(f.getXDim());
            s = 0;
            r = 0;

            rho1 = 0.9;
            rho2 = 0.999;
            rho1_product = 1;
            rho2_product = 1;
            epsilon = 0.001;

            delta = MATH21_10NEG7;
            x.setSize(f.getXDim());
            x = 0;
            tmp.setSize(f.getXDim());
            y = f.valueAt(x);
        }

        void update() {
            const VecR &g = f.derivativeValueAt(x);
            rho1_product *= rho1;
            rho2_product *= rho2;

            math21_operator_linear_to_A(rho1, s, (1 - rho1), g);
            math21_operator_SchurProduct(g, g, tmp);
            math21_operator_linear_to_A(rho2, r, (1 - rho2), tmp);
            math21_operator_linear((1 / (1 - rho1_product)), s, s_hat);
            math21_operator_linear((1 / (1 - rho2_product)), r, r_hat);

            VecR &tmp_2 = tmp;
            math21_operator_add(delta, r_hat, tmp_2);
            math21_operator_sqrt_to(tmp_2);
            math21_operator_divide_to(-epsilon, tmp_2);

            VecR &delta_x = tmp_2;
            math21_operator_SchurProduct_to_A(delta_x, s_hat);

            math21_operator_addToA(x, delta_x);

            y_old = y;
            y = f.valueAt(x);
        }
    };

    /*
     * min f(x)
     *
     * */
    class SteepestDescent : public think::Optimization {
    private:
        sd_update_rule &update_rule;
        OptimizationInterface &oi;
    public:
        SteepestDescent(sd_update_rule &update_rule, OptimizationInterface &optimizationInterface);

        virtual ~SteepestDescent() {}

        void solve();

        Functional &getFunctional();

        NumN getTime();

        const VecR &getMinimum() { return update_rule.x; }
    };
}