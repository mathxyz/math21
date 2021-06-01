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

#include "../ad/functions_01/ad_global.h"
#include "Variable.h"
#include "VariableMap.h"

namespace math21 {
    namespace ad {

        NumN VariableMap::_createSharedC(NumR x, const char *name) {
            global.push(new Variable());
            NumN sharedId = global.size();
            Variable &vk = *global.at(sharedId);
            vk.setId(0);
            vk.setName(name);
            vk.setType(variable_type_constant);
            vk.setf(ad_global_get_op_num_constant());
            vk.getValue().setSize(1);
            vk.getValue() = x;
            return sharedId;
        }

        void VariableMap::_createSomeSharedC() {
            constant_0 = _createSharedC(0, "0");
            constant_1 = _createSharedC(1, "1");
            constant_m1 = _createSharedC(-1, "-1");
        }

        void VariableMap::init() {
            ad_global_create(*this);
            _createSomeSharedC();
            local_size_backup = 0;
            global_size_backup = 0;
            v_size_backup = 0;
        }

        void VariableMap::backup() {
            MATH21_ASSERT(!Variable::isHavingY(), "not implement")
            // no need to backup V when cd_inc
            // Because the time cost is little, we don't optimize it.
            MATH21_PRINT_TIME_ELAPSED(V_backup.set(V));
            local_size_backup = local.size();
            global_size_backup = global.size();
            v_size_backup = v.size();
        }

        void VariableMap::reset() {
            clear();
            init();
        }

        void VariableMap::restore() {
            if (isEmpty()) {
                return;
            }
            MATH21_ASSERT(!Variable::isHavingY(), "not implement")
            MATH21_ASSERT(V.size() >= V_backup.size());
            MATH21_ASSERT(local.size() >= local_size_backup);
            MATH21_ASSERT(global.size() >= global_size_backup);
            MATH21_ASSERT(v.size() >= v_size_backup);
            V.set(V_backup);
            for (NumN i = local_size_backup + 1; i <= local.size(); ++i) {
                delete local.at(i);
            }
            for (NumN i = global_size_backup + 1; i <= global.size(); ++i) {
                delete global.at(i);
            }
            math21_operator_container_sub_from_start(local, local_size_backup);
            math21_operator_container_sub_from_start(global, global_size_backup);
            math21_operator_container_sub_from_start(v, v_size_backup);
        }

        void VariableMap::clear() {
            // It had better be called at last, but can still put here.
            ad_global_destroy();

            for (NumN i = 1; i <= local.size(); ++i) {
                delete local.at(i);
            }
            for (NumN i = 1; i <= global.size(); ++i) {
                delete global.at(i);
            }

            local.clear();
            global.clear();
            V.clear();
            v.clear();

            constant_0 = 0;
            constant_1 = 0;
            constant_m1 = 0;

            V_backup.clear();
            local_size_backup = 0;
            global_size_backup = 0;
            v_size_backup = 0;
        }

        NumN VariableMap::get_constant_0() {
            return _createV("", 1, constant_0);
        }

        NumN VariableMap::get_constant_1() {
            return _createV("", 1, constant_1);
        }

        NumN VariableMap::get_constant_m1() {
            return _createV("", 1, constant_m1);
        }

        VariableMap::VariableMap(Set &V) : V(V) {
            init();
        }

        VariableMap::~VariableMap() {
            clear();
        }

        NumN VariableMap::size() const {
            return v.size();
        }

        NumB VariableMap::isEmpty() const {
            if (size() == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        NumB VariableMap::log(const char *s) const {
            v.log(s);
            return 1;
        }

        NumB VariableMap::log(std::ostream &io, const char *s) const {
            v.log(io, s);
            return 1;
        }

        NumN VariableMap::_createV(const char *name, NumB isShared, NumN sharedId) {
            if (isShared) {
                v.push(global.at(sharedId));
                NumN id = v.size();
                V.add(id);
            } else {
                local.push(new Variable());
                v.push(local.getLast());

                NumN id = v.size();
                Variable &x2 = *v.getLast();
                x2.setId(id);
                x2.setName(name);
                V.add(id);
            }
            NumN id = v.size();
            if (id == math21_global_ad_debug_var_id()) {
                MATH21_ASSERT(0)
            }
//            m21log("variable id", id);
            return id;
        }

        // this will make previous references to pointer content invalid.
        // So use references to data.
        NumN VariableMap::createV(const char *name) {
            NumN k = _createV(name, 0, 0);
            Variable &vk = at(k);
            return k;
        }

        // create constant.
        NumN VariableMap::createC(const char *name) {
            NumN k = _createV(name, 0, 0);
            Variable &vk = at(k);
            vk.setType(variable_type_constant);
            return k;
        }

        void VariableMap::setDeviceType(NumN id, NumN deviceType) {
            Variable &vk = at(id);
            vk.getValue().setDeviceType(deviceType);
        }

        void VariableMap::setValue(NumN id, NumR x) {
            Variable &vk = at(id);
            vk.getValue().setSize(1);
            vk.getValue() = x;
        }

        Variable &VariableMap::at(NumN i) {
//            if (i == 708) {
//                int aaa=1;
//            }
            return *v.at(i);
        }

        const Variable &VariableMap::operator()(NumN i) const {
//            if (i == 708) {
//                int aaa=1;
//            }
            return *v.operator()(i);
        }

        Set &VariableMap::getV() {
            return V;
        }

        // constant X by variable X
        void setSizeCXUXByX(const Set &X, VariableMap &data) {
            NumN x = 0;
            for (NumN i = 1; i <= X.size(); ++i) {
                NumN type = data.at(X(i)).getType();
                if (type != variable_type_constant) {
                    x = X(i);
                    // x is used
                    if (!data.at(x).getValue().isEmpty()) {
                        break;
                    }
                }
            }
            if (x != 0) {
                for (NumN i = 1; i <= X.size(); ++i) {
                    NumN type = data.at(X(i)).getType();
                    if (type == variable_type_constant) {
                        Variable &vk = data.at(X(i));
                        if (vk.getValue().isScalarInMath()) {
                            if (data.at(x).getValue().isScalarInMath()) {
                                continue;
                            }
                            NumR c = vk.getValue()(1);
                            vk.getValue().setSize(data.at(x).getValue().shape());
                            vk.getValue() = c;
                        }
                    } else {
                        if (data.at(X(i)).getValue().isEmpty()) {
                            data.at(X(i)).getValue().setSize(data.at(x).getValue().shape());
                            data.at(X(i)).getValue() = 0;
                        }
                    }
                }
            }
        }

        // constant Y by X
        void setSizeYByX(const Set &X, const Set &Y, VariableMap &data) {
            const ArrayN &d = data.at(X(1)).getValue().shape();
            for (NumN i = 1; i <= Y.size(); ++i) {
                NumN y = Y(i);
                data.at(y).getValue().setSize(d);
            }
        }

        // set size of y by x
        void _setSizeyByx(NumN x, NumN y, VariableMap &data) {
            const ArrayN &d = data.at(x).getValue().shape();
            if (!data.at(y).getValue().isSameSize(d)) {
                data.at(y).getValue().setSize(d);
            }
        }

        // set size of y by x
        void _setSizeCyByx(NumN x, NumN y, VariableMap &data) {
            const ArrayN &d = data.at(x).getValue().shape();
            if (!data.at(y).getValue().isSameSize(d)) {
                Variable &vk = data.at(y);
                NumR c = 0;
                if (vk.getValue().isScalarInMath()) {
                    c = vk.getValue()(1);
                }
                vk.getValue().setSize(d);
                vk.getValue() = c;
            }
        }

        // set size of y by x
        void setSizeyByx(NumN x, NumN y, VariableMap &data) {
            NumN type = data.at(y).getType();
            if (type == variable_type_constant) {
                _setSizeCyByx(x, y, data);
            } else {
                _setSizeyByx(x, y, data);
            }
        }
    }
}