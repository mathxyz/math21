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
        enum {
            variable_type_default = 1,
            variable_type_input,
            variable_type_output,
            variable_type_constant, // variable, but has constant value.
            variable_type_list, // variable. Its elements are also variables.
            variable_type_zero_derivative, // variable. Its elements are also variables.
        };

        const char *math21_type2string_variable(NumN variable_type);

        // variable content class
        struct Variable {
        private:
            std::string name;
            // shared variable has more than one id, so the id is set to 0 to disable using id.
            // If must use id, then use map<id, var>.
            // So use variable id other than variable content to denote a variable and discriminate between two variables.
            NumN id;
            NumN type;
            Function *f;
            // For y = f(x), dy = f'(x), suppose Y = {dy},
            // if dy depends on value of y, then X = {x, y}
            // if not, then X = {x}.
            Set X;
            Set Y;
            // 0: matrix is composed as every elements. No abstraction. This mode makes computing higher differential easier.
            // 1: matrix is seen as a whole one. Abstract completely.
            // 2: Abstract partially. non-trivial block matrix not supported currently.
            NumN abstractLevel;
            static NumN requestedAbstractLevel;
            TenN variableMat; // variable matrix, i.e., every element is a variable id pointing to the variable in data.
            TenR v; // when isDense
            // if not dense, then must be generated by f
            // and f is responsible for converting it to dense.
            NumB _isDense;
            NumB _computed;
            NumB _is_f_cloned;

            // temporary
            Set cacheY;

            void init();

            void copy(const Variable &v);

            void setAbstractLevel(NumN level);

            static NumB _hasY;
        public:

            static void setHasY(NumB hasY) {
                _hasY = hasY;
            }

            static NumB isHavingY() {
                return _hasY;
            }

            static void setRequestedAbstractLevel(NumN level);

            static NumB isRequestingAbstractZero();

            static NumB isRequestingAbstractCompletely();

            void setAbstractZero();

            NumB isAbstractZero() const;

            NumB isAbstractCompletely() const;

            Variable();

            Variable(const Variable &v);

//            Variable(Function *f);

            virtual ~Variable();

            TenN &getVariableMat();

            const TenN &getVariableMat() const;

            void addx(NumN x);

            void addX(const Set &X0);

            void setX(const Set &X0);

            Set &getX();

            const Set &getX() const;

            void addy(NumN y);

            void add_cache_y(NumN y);

//            Set &getY();

            TenR &getValue();

            const TenR &getValue() const;

            const Set &getY() const;

            const Set &getCacheY() const;

            void clearCacheY();

            void setf(Function *f);

            Function &getf();

            NumB hasf() const;

            NumB isDense() const;

            void setDense(NumB dense);

            const Function &getf() const;

            void setId(NumN id);

//            NumN getId() const;

            const std::string &getName() const;

            void setName(const char *name = 0);

            NumN getType() const;

            void setType(NumN type);

            void log(const char *name2 = 0, NumB isLogDetail = 0) const;

            void log(std::ostream &io, const char *name2 = 0, NumB isLogDetail = 0) const;

            void reset();

            NumB isComputed() const;

            void setComputed(NumB computed);

            void synchronizeValue(VariableMap &data);

            void synchronizeToZero(VariableMap &data);
        };

        NumB ad_is_containing_constant_num_0(const Set &X, VariableMap &data);

        NumB ad_is_constant_num(NumN x, const VariableMap &data);
    }
}