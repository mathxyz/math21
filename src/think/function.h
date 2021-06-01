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
    namespace ad {
        struct VariableMap;

        class Derivative;

        // used as Function input and output
        struct AdVar {
        private:
            NumN id;
            VariableMap *data;
        public:
            AdVar();

            AdVar(NumN id, VariableMap *data);

            AdVar(const AdVar &p);

            virtual ~AdVar();

            AdVar &operator=(const AdVar &p);
        };

        // used as Function inputs
//        struct AdVars {
//        private:
//            NumN id;
//            VariableMap *data;
//        public:
//            AdVars();
//
//            AdVars(NumN id, VariableMap *data);
//
//            AdVars(const AdVars &p);
//
//            virtual ~AdVars();
//
//            AdVars &operator=(const AdVars &p);
//        };

        struct Function {
        private:
//            static const NumB isSetSizeFlag;
            // deprecate
            static NumB isSetSizeFlag;
            // type 1: element-wise
            // type 2: not element-wise
            static NumB isElementWiseTestFlag;
            NumB isElementWiseFlag;
            NumB isGlobalFlag;


            virtual void cr_jvp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const;

            // We decide to reshape dx at end.
            // todo: reshape dx at end, or reshape dy at start.
            // todo: if we reshape dx at end, y can be left out.
            // see autograd/numpy/numpy_vjps.py
            // 'return dx = 0' means differential being zero.
            // chain rule in define-by-run mode
            virtual NumN cr_vjp_inner(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const;

            virtual void cr_jmp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const;

            virtual void cr_mjp(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const;

        public:
            Function();

            virtual ~Function();

            // todo: maybe remove virtual
            virtual NumN cr_vjp(const Set &X, NumN x, NumN y, NumN dy, VariableMap &data) const;

            NumN evaluate(NumN x, VariableMap &data);

            NumN evaluate(NumN x1, NumN x2, VariableMap &data);

            NumN evaluate(NumN x1, NumN x2, NumN x3, VariableMap &data);

            // todo: implement here instead of in every function.
            // define-by-run <=> f + fv
            virtual NumN evaluate(const Set &X, VariableMap &data);

            void f(NumN x, NumN &y, VariableMap &data);

            void f(NumN x1, NumN x2, NumN &y, VariableMap &data);

            void f(NumN x1, NumN x2, NumN x3, NumN &y, VariableMap &data);

            void compute(NumN x, NumN y, VariableMap &data, Derivative &derivative);

            void compute(NumN x1, NumN x2, NumN y, VariableMap &data, Derivative &derivative);

            void compute(NumN x1, NumN x2, NumN x3, NumN y, VariableMap &data, Derivative &derivative);

            void forward(NumN x, NumN &y, VariableMap &data);

            void forward(NumN x1, NumN x2, NumN &y, VariableMap &data);

            void forward(NumN x1, NumN x2, NumN x3, NumN &y, VariableMap &data);

            // may deprecate, use Derivative.cd() instead.
            virtual void df(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const;

            // df in define-by-run mode
            virtual void df_dbr(const Set &X, NumN x, NumN y, NumN &dydx, VariableMap &data) const;

            // chain rule
            virtual void cr(const Set &X, NumN x, NumN y, NumN dy, Set &Y, VariableMap &data) const;

            // deprecate, use cr_vjp
            // chain rule in define-by-run mode
            virtual void backward(const Set &X, NumN x, NumN y, NumN dy, Set &output, VariableMap &data) const;

            // define function graph
            virtual void f(const Set &X, Set &Y, VariableMap &data);

            // run function graph
            // evaluate function value
            virtual void fv(const Set &X, const Set &Y, VariableMap &data) const;

            // run in partial define-by-run mode
            // evaluate function value, modify graph if necessary.
            virtual void compute(const Set &X, const Set &Y, VariableMap &data, Derivative &derivative);

            // define-by-run <=> f + fv
            virtual void forward(const Set &X, Set &Y, VariableMap &data);

            // deprecate
            virtual void setSize(const Set &X, const Set &Y, VariableMap &data) const {}

            virtual Function *clone() const = 0;

            virtual const char *getName() const = 0;

            // deprecated, use compute instead.
            static NumB isSetSize();

            static void setSetSizeFlag(NumB flag);

            // test
            static NumB isElementWiseTest();

            NumB isElementWise() const;

            void setElementWiseFlag(NumB flag);

            NumB isGlobal() const;

            void setGlobalFlag(NumB flag);

            static void broadcast_tensors(const Set &X, Set &Y, VariableMap &data);

            static void broadcast_num_to_vec(const Set &X, Set &Y, VariableMap &data);

            static void variable_set_device_type_using_variable(NumN x, NumN y, VariableMap &data);

            static void variable_set_device_type_gpu(NumN y, VariableMap &data);

            static NumN variable_get_device_type(NumN x, VariableMap &data);

            static NumB variable_is_cpu(NumN x, VariableMap &data);

            static NumB variable_setSize_to_same_vspace_using_variable(NumN x, NumN y, VariableMap &data);

            static NumB variable_reshape_to_same_vspace_using_variable(NumN x, NumN y, VariableMap &data);

            static NumB variable_setSize_to_same_vspace_using_shape(const VecN &d, NumN y, VariableMap &data);

            static NumB variable_reshape_to_same_vspace_using_shape(const VecN &d, NumN y, VariableMap &data);
        };
    }
}