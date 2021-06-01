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

    class LogOperator : public think::Operator {
    private:
    public:
        LogOperator() {
        }

        virtual ~LogOperator() {}

        static void valueAt(const TenR &x, TenR &y) {
            math21_operator_log(x, y);
        }
    };

    class ExpOperator : public think::Operator {
    private:
    public:
        ExpOperator() {
        }

        virtual ~ExpOperator() {}

        static void valueAt(const TenR &x, TenR &y) {
            math21_operator_exp(x, y);
        }
    };

    class SoftargmaxOperator : public think::Operator {
    public:
        SoftargmaxOperator() {
        }

        virtual ~SoftargmaxOperator() {}

        static void valueAt(const TenR &x, TenR &y) {
            ExpOperator::valueAt(x, y);
            NumR sum = math21_operator_norm(y, 1);
            MATH21_ASSERT(sum > 0,
                          "norm as denominator which should be larger than 0, but it is "
                                  << sum << "\n"
                                  << "\t" << x.log("x") << "\n"
                                  << "\t" << y.log("y") << "\n"
            )
//            math21_operator_clip_not_less_than_eps(sum);
            math21_operator_linear_to(1 / sum, y);
            MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_is_not_less(y, 0))
        }
    };

    class PureLinearOperator : public think::Operator {
    public:
        PureLinearOperator() {
        }

        virtual ~PureLinearOperator() {}

//        void valueAt(const VecR &x, VecR &y) {
//            NumN n = x.size();
//            if (y.size() != n) {
//                y.setSize(n);
//            }
//            y.assign(x);
//        }

        void derivativeValueAt(const VecR &x, MatR &dH) {
            if (dH.nrows() != x.size() || dH.ncols() != x.size()) {
                dH.setSize(x.size(), x.size());
            }
            math21_operator_mat_eye(dH);
        }

        static void derivativeValueUsingf(const VecR &y, MatR &dH) {
            if (dH.isSameSize(y.size(), y.size()) == 0) {
                dH.setSize(y.size(), y.size());
            }
            math21_operator_mat_eye(dH);
        }
    };

    class CostFunctional_class : public Functional {
    public:
        CostFunctional_class() {}

        virtual ~CostFunctional_class() {}

        virtual void setParas(const VecR &_t) = 0;

//        virtual void setParas(const TenR &_t) = 0;

        virtual void clear() = 0;
    };

    //Todo: add tensor support.
    class CostFunctional_mse_se_class : public CostFunctional_class {
    private:
        NumB isSet;
        VecR t;
        VecR g;
    public:
        CostFunctional_mse_se_class() {
            isSet = 0;
        }

        virtual ~CostFunctional_mse_se_class() {}

        NumR valueAt(const VecR &x) override {
            MATH21_ASSERT(isSet, "Please set parameters first!")
            NumR fx;
            VecR tmp;
            math21_operator_vec_linear(1, t, -1, x, tmp);
            fx = math21_operator_InnerProduct(1, tmp, tmp);
            return fx;
        }

        NumN getXDim() override {
            MATH21_ASSERT(isSet, "Please set parameters first!")
            return t.size();
        }

        const VecR &derivativeValueAt(const VecR &x) override {
            math21_operator_vec_linear(2, x, -2, t, g);
            return g;
        }

        void setParas(const VecR &_t) override {
            if (t.isSameSize(_t.size()) == 0) {
                t.setSize(_t.size());
            }
            t.assign(_t);
            isSet = 1;
        }

        void clear() override{
            isSet = 0;
        }
    };

    class CostFunctional_nll_CrossEntroy_softmax_class : public CostFunctional_class {
    private:
        NumB isSet;
        TenR t;
        TenR g;
        TenR y_softmax;

        VecR tmp;
    public:
        CostFunctional_nll_CrossEntroy_softmax_class() {
            isSet = 0;
        }

        virtual ~CostFunctional_nll_CrossEntroy_softmax_class() {}

        NumR valueAt(const TenR &x) override {
            MATH21_ASSERT(isSet, "Please set parameters first!")
            NumR fx;
            SoftargmaxOperator::valueAt(x, y_softmax);
            math21_operator_clip_not_less_than_eps(y_softmax);
            LogOperator::valueAt(y_softmax, tmp);
            MATH21_ASSERT_CHECK_VALUE_TMP(math21_operator_is_not_larger(tmp, 0))
            fx = math21_operator_InnerProduct(-1, tmp, t);
            return fx;
        }

        void setParas(const TenR &_t) override {
            t.setSize(_t.shape());
            t.assign(_t);
            isSet = 1;
        }

        void clear()override {
            isSet = 0;
        }

        const TenR &derivativeValueAt(const TenR &x) override {
            valueAt(x);
            math21_operator_linear(1, y_softmax, -1, t, g);
            return g;
        }


        // Todo: maybe getXshape
        NumN getXDim() override{
            MATH21_ASSERT(isSet, "Please set parameters first!")
            return t.size();
        }

        const ArrayN &get_x_shape() const override{
            return t.shape();
        }

    };

    class dummy_class {
    public:
        dummy_class() {
        }

        virtual ~dummy_class() {}
    };

    ////####################################

    class Function_linear : public Function {
    private:
    public:
        Function_linear() {
        }

        virtual ~Function_linear() {}

        NumR valueAt(const NumR &x) override{
            return x;
        }

        NumR derivativeValueAt(const NumR &x) override {
            return 1;
        }

        NumR derivativeValue_using_y(const NumR &y) override {
            return 1;
        }
    };

    class Function_tanh : public Function {
    private:
    public:
        Function_tanh() {
        }

        virtual ~Function_tanh() {}

        NumR valueAt(const NumR &x) override{
            NumR y;
            NumR a, b;
            a = xjexp(x);
            b = xjexp(-x);
            y = (a - b) / (a + b);
            return y;
        }

        NumR derivativeValueAt(const NumR &x) override {
            NumR y;
            y = valueAt(x);
            return 1 - y * y;
        }

        NumR derivativeValue_using_y(const NumR &y) override {
            return 1 - y * y;
        }
    };

    class Function_LogSigmoid : public Function {
    private:
    public:
        Function_LogSigmoid() {
        }

        virtual ~Function_LogSigmoid() {}

        NumR valueAt(const NumR &x) override{
            return 1.0 / (1 + xjexp(-x));
        }

        NumR derivativeValueAt(const NumR &x) override {
            NumR y = valueAt(x);
            return (1 - y) * y;
        }

        NumR derivativeValue_using_y(const NumR &y) override {
            return (1 - y) * y;
        }
    };

    class Function_LeakyReLU : public Function {
    private:
    public:
        Function_LeakyReLU() {
        }

        virtual ~Function_LeakyReLU() {}

        NumR valueAt(const NumR &x) override {
            if (x >= 0) {
                return x;
            } else {
                return 0.01 * x;
            }
        }

        NumR derivativeValueAt(const NumR &x) override {
            if (x >= 0) {
                return 1;
            } else {
                return 0.01;
            }
        }

        NumR derivativeValue_using_y(const NumR &y) override {
            if (y >= 0) {
                return 1;
            } else {
                return 0.01;
            }
        }
    };

    ////####################################

    ////!!!! deprecated
    class LogSigmoid : public Function {
    private:
    public:
        LogSigmoid() {
        }

        virtual ~LogSigmoid() {}

        NumR valueAt(const NumR &x) override {
            return 1.0 / (1 + xjexp(-x));
        }

        NumR derivativeValueAt(const NumR &x) override {
            NumR y = valueAt(x);
            return (1 - y) * y;
        }

        static NumR derivativeValueUsingf(const NumR &y) {
            return (1 - y) * y;
        }
    };

    inline NumR m21tanh(const NumR &x) {
        NumR y;
        NumR a, b;
        a = xjexp(x);
        b = xjexp(-x);
        y = (a - b) / (a + b);
        return y;
    }

    inline NumR m21tanh_derivativeValueUsing_y(const NumR &y) {
        return 1 - y * y;
    }

    ////!!!! deprecated
    class LeakyReLU : public Function {
    private:
    public:
        LeakyReLU() {
        }

        virtual ~LeakyReLU() {}

        NumR valueAt(const NumR &x) override {
            if (x >= 0) {
                return x;
            } else {
                return 0.01 * x;
            }
        }

        NumR derivativeValueAt(const NumR &x) override {
            if (x >= 0) {
                return 1;
            } else {
                return 0.01;
            }
        }

        static NumR derivativeValueUsingf(const NumR &y) {
            if (y >= 0) {
                return 1;
            } else {
                return 0.01;
            }
        }
    };

    class LogSigmoidOperator : public think::Operator {
    private:
    public:
        LogSigmoidOperator() {
        }

        virtual ~LogSigmoidOperator() {}

        void valueAt(const VecR &x, VecR &y) {
            LogSigmoid ls;
            y.setSize(x.size());
            for (NumN i = 1; i <= x.size(); i++) {
                y(i) = ls.valueAt(x(i));
            }
        }

        void derivativeValueAt(const VecR &x, MatR &dH) {
            dH.setSize(x.size(), x.size());
            dH = 0;
            LogSigmoid ls;
            for (NumN i = 1; i <= x.size(); i++) {
                dH(i, i) = ls.derivativeValueAt(x(i));
            }
        }

        static void derivativeValueUsing_y(const VecR &y, MatR &dH) {
            dH.setSize(y.size(), y.size());
            dH = 0;
            for (NumN i = 1; i <= y.size(); i++) {
                dH(i, i) = LogSigmoid::derivativeValueUsingf(y(i));
            }
        }
    };

    class Operator_tanh : public think::Operator {
    private:
    public:
        Operator_tanh() {
        }

        virtual ~Operator_tanh() {}

        void valueAt(const VecR &x, VecR &y) {
            y.setSize(x.size());
            for (NumN i = 1; i <= x.size(); i++) {
                y(i) = m21tanh(x(i));
            }
        }

        void derivativeValueAt(const VecR &x, MatR &dH) {
            MATH21_ASSERT_NOT_CALL(0, "dummy");
            dH.setSize(x.size(), x.size());
            dH = 0;
        }

        static void derivativeValueUsing_y(const VecR &y, MatR &dH) {
            dH.setSize(y.size(), y.size());
            dH = 0;
            for (NumN i = 1; i <= y.size(); i++) {
                dH(i, i) = m21tanh_derivativeValueUsing_y(y(i));
            }
        }
    };

    class LeakyReLUOperator : public think::Operator {
    public:
        LeakyReLUOperator() {
        }

        virtual ~LeakyReLUOperator() {}

        VecR valueAt(const VecR &x) {
            LeakyReLU ls;
            VecR y(x.size());
            for (NumN i = 1; i <= x.size(); i++) {
                y(i) = ls.valueAt(x(i));
            }
            return y;
        }

        void derivativeValueAt(const VecR &x, MatR &dH) {
            dH.setSize(x.size(), x.size());
            dH = 0;
            LeakyReLU ls;
            for (NumN i = 1; i <= x.size(); i++) {
                dH(i, i) = ls.derivativeValueAt(x(i));
            }
        }

        static void derivativeValueUsing_y(const VecR &y, MatR &dH) {
            dH.setSize(y.size(), y.size());
            dH = 0;
            for (NumN i = 1; i <= y.size(); i++) {
                dH(i, i) = LeakyReLU::derivativeValueUsingf(y(i));
            }
        }
    };

    class FunctionSineEx1 : public Function {
    private:
    public:
        FunctionSineEx1() {
        }

        virtual ~FunctionSineEx1() {}

        NumR valueAt(const NumR &x) override {
            return 1 + xjsin((XJ_PI / 4.0) * x);
        }

        NumR derivativeValueAt(const NumR &x) override {
            MATH21_ASSERT_NOT_CALL(0, "can't call.");
            return x;
        }
    };
}