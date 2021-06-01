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
        // how to simplify a graph?
        // 1 graph level for a target function.
        // It happens between different ops.
        // If a graph is representing a specific function,
        // then the graph can be simplified.
        // If isn't, then it can't be simplified.
        // 2 op level simplification.
        // It happens within the op and this is independent of the graph.

        // abstract level
        // absolute abstraction: fast
        // zero abstraction: slow

        // mode 1: define-and-run
        // mode 2: define, then run modifying graph when necessary.
        // mode 3: define-by-run
        enum {
            derivative_mode_dar = 1, // define-and-run
            derivative_mode_dar_dbr, // define, then run modifying graph when necessary.
            derivative_mode_dbr, // define-by-run
            derivative_mode_dbr_jvp, // in define-by-run
            derivative_mode_dbr_vjp,
            derivative_mode_dbr_jmp,
            derivative_mode_dbr_mjp,
        };

        // todo: destroy graph node immediately.
        class Derivative {
        private:
            NumB _is_cd_inc;
            VariableMap &data;
            NumN debugLevel;

            void fv(const Set &X, NumN y, NumN mode, NumN level = 1);

            // Chapter 6 of book 'Deep Learning' by Ian Goodfellow, (http://www.deeplearningbook.org/)
            // DT is derivative table of y respect to all related variables.
            // if y is different, DT should be reset.
            void _cds(const Set &X, NumN y, const Set &V, Map &dX, Map &DT, NumN mode, NumN n);

            // Refer to backward_pass in autograd/core.py for details. [Autograd](https://github.com/HIPS/autograd)
            // _cds <=> _cds_inc
            void _cds_inc(const Set &X, NumN y, const Set &V, Map &dX, Map &DT, NumN mode, NumN n);

            NumN grad_with_mode(NumN x, NumN y, NumN mode);

        public:

            Derivative(VariableMap &data);

            virtual ~Derivative();

            // if never played, i.e., f' not generated, then function f is removed instead.
            // if f'' is generated, then f'' is removed.
            void removeLastRecord();

            NumN cd(NumN x, NumN y);

            NumN backward(NumN x, NumN y);

            NumN grad_jvp(NumN x, NumN y);

            NumN grad_vjp(NumN x, NumN y);

            NumN cd(NumN x, NumN y, Map &DT);

            // compute derivative. Set value to constant. n is used for debug.
            void cds(const Set &X, NumN y, const Set &V, Map &dX, NumN n = 1);

            // compute function values where no variable size in program level will be considered.
            void fvs(const Set &X, const Set &Y);

            void compute(NumN y);

            void compute(const Set &X, NumN y);

            void compute(const Set &X, const Set &Y, const Set &V, NumB isReset = 1);

            // make graph simple when a function is given.
            void fuse() {

            }

            void setDebugLevel(NumN debugLevel0);

            VariableMap &getData();
        };

        void math21_ad_diff_cd_inc_with_mode_dbr_vjp(
                const Set &X, NumN y, Map &DT, NumN mode, VariableMap &data);

    }
}