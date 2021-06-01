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

#include "VariableMap.h"
#include "Variable.h"
#include "../ad/files.h"
#include "../ad/functions_01/files.h"
#include "../op/files.h"

namespace math21 {
    namespace ad {
        const char *math21_type2string_variable(NumN variable_type) {
#define MATH21_LOCAL_F(a) case a: return MATH21_STRINGIFY(a);
            switch (variable_type) {
                MATH21_LOCAL_F(variable_type_default)
                MATH21_LOCAL_F(variable_type_input)
                MATH21_LOCAL_F(variable_type_output)
                MATH21_LOCAL_F(variable_type_constant)
                default:
                    return "UNKNOWN";
            }
#undef MATH21_LOCAL_F
        }

        NumB Variable::_hasY = 0;
//        NumB Variable::_hasY = 1;

        NumN Variable::requestedAbstractLevel = 0;

        void Variable::setRequestedAbstractLevel(NumN level) {
            MATH21_ASSERT(level <= 1)
            requestedAbstractLevel = level;
        }

        NumB Variable::isRequestingAbstractZero() {
            if (requestedAbstractLevel == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB Variable::isRequestingAbstractCompletely() {
            if (requestedAbstractLevel == 1) {
                return 1;
            } else {
                return 0;
            }
        }

        void Variable::init() {
            id = 0;
            f = 0;
            type = variable_type_default;
            _isDense = 1;
            _computed = 0;
            abstractLevel = 1; // must initialize to 1
            _is_f_cloned = 0;
        }

        void Variable::copy(const Variable &v) {
            printf("Variable copy, error!\n");
        }

        void Variable::setAbstractLevel(NumN level) {
            MATH21_ASSERT(level <= 1)
            abstractLevel = level;
        }

        void Variable::setAbstractZero() {
            MATH21_ASSERT(!isAbstractZero())
            setAbstractLevel(0);
        }

        NumB Variable::isAbstractZero() const {
            if (abstractLevel == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB Variable::isAbstractCompletely() const {
            if (abstractLevel == 1) {
                return 1;
            } else {
                return 0;
            }
        }

        Variable::Variable() {
            init();
        }

        Variable::Variable(const Variable &v) {
            init();
            copy(v);
        }

//        Variable::Variable(Function *f) {
//            init();
//            if(f.isGlobal()){}else{}
//            this->f = f->clone();
//        }

        Variable::~Variable() {
            if (f) {
                if (_is_f_cloned) {
                    delete f;
                }
                f = 0;
            }
        }

        TenN &Variable::getVariableMat() {
            MATH21_ASSERT(abstractLevel < 1)
            return variableMat;
        }

        const TenN &Variable::getVariableMat() const {
            MATH21_ASSERT(abstractLevel < 1)
            return variableMat;
        }

        void Variable::addx(NumN x) {
            X.add(x);
        }

        void Variable::addX(const Set &X0) {
            X.add(X0);
        }

        void Variable::setX(const Set &X0) {
            X0.copyTo(X);
        }

        Set &Variable::getX() {
            return X;
        }

        const Set &Variable::getX() const {
            return X;
        }

        void Variable::addy(NumN y) {
            Y.add(y);
        }

        void Variable::add_cache_y(NumN y) {
            cacheY.add(y);
        }

//        Set &Variable::getY() {
//            return Y;
//        }

        TenR &Variable::getValue() {
            MATH21_ASSERT(isDense());
            return v;
        }

        const TenR &Variable::getValue() const {
            MATH21_ASSERT(isDense());
            return v;
        }

        const Set &Variable::getY() const {
            MATH21_ASSERT(isHavingY(), "You must enable Y by calling setHasY()!");
            return Y;
        }

        const Set &Variable::getCacheY() const {
            return cacheY;
        }

        void Variable::clearCacheY() {
            cacheY.clear();
        }

        void Variable::setf(Function *f) {
            if(f->isGlobal()){
                this->f = f;
                _is_f_cloned = 0;
            }else{
                this->f = f->clone();
                _is_f_cloned = 1;
            }
        }

        Function &Variable::getf() {
            return *f;
        }

        NumB Variable::hasf() const {
            if (f == 0) {
                return 0;
            }
            return 1;
        }

        NumB Variable::isDense() const {
            return _isDense;
        }

        void Variable::setDense(NumB dense) {
            _isDense = dense;
        }

        const Function &Variable::getf() const {
            if (f == 0) {
                printf("f = 0");
            }
            return *f;
        }

        void Variable::setId(NumN id) {
            this->id = id;
        }

//        NumN Variable::getId() const {
//            return id;
//        }

        const std::string &Variable::getName() const {
            return name;
        }

        void Variable::setName(const char *name) {
            if (name) {
                this->name = name;
            } else {
                this->name = "";
            }
        }

        NumN Variable::getType() const {
            return type;
        }

        void Variable::setType(NumN type) {
            this->type = type;
        }

        void Variable::log(const char *name2, NumB isLogDetail) const {
            log(std::cout, name2, isLogDetail);
        }

        void Variable::log(std::ostream &io, const char *name2, NumB isLogDetail) const {
            io << "id: " << id << ", ";
            io << "name: " << (name2 == 0 ? name.c_str() : name2) << ", ";
            io << "type: " << math21_type2string_variable(type) << "\n";
            X.log(io, "input");
            Y.log(io, "output");
            if (f) {
                io << "f name: " << f->getName();
//                io << ", f addr: " << f;
                io << "\n";
            }
            v.log(io, "value");
            if(math21_global_ad_is_check_nan()){
                NumB flag = math21_op_check_is_nan(v);
                if(flag){
                    MATH21_ASSERT(0)
                }
            }
            if (isAbstractZero() && isLogDetail) {
                variableMat.log(io, "variableMat");
            }
        }

        void Variable::reset() {
            _computed = 0;
        }

        // called by diff, not by function
        NumB Variable::isComputed() const {
            return _computed;
        }

        void Variable::setComputed(NumB computed) {
            _computed = computed;
        }

        // synchronizeToWhole
        void Variable::synchronizeValue(VariableMap &data) {
            if (isAbstractZero()) {
                v.setSize(variableMat.shape());
                for (NumN i = 1; i <= variableMat.size(); ++i) {
                    NumN x = variableMat(i);
                    v.at(i) = data(x).getValue()(1);
                }
            }
        }

        void Variable::synchronizeToZero(VariableMap &data) {
            if (isAbstractCompletely()) {
                variableMat.setSize(v.shape());
                for (NumN i = 1; i <= variableMat.size(); ++i) {
                    NumN x = data.createV(math21_string_concatenate(name, math21_string_to_string(i)).c_str());
                    Variable &vx = data.at(x);
                    vx.setf(ad_global_get_op_num_input());
                    vx.setType(type);
                    vx.getValue().setSize(1);
                    vx.getValue() = v(i);
                    variableMat.at(i) = x;
                }
                setAbstractZero();
            }
        }

        NumB ad_is_containing_constant_num_0(const Set &X, VariableMap &data) {
            for (NumN i = 1; i <= X.size(); ++i) {
                const auto &v = data(X(i));
                if (v.getType() == variable_type_constant) {
                    if (v.getValue().size() == 1) {
                        if (v.getValue()(1) == 0) {
                            return 1;
                        }
                    }
                }
            }
            return 0;
        }

        NumB ad_is_constant_num(NumN x, const VariableMap &data) {
            const auto &v = data(x);
            if (v.getType() == variable_type_constant) {
                return 1;
            }
            return 0;
        }
    }
}