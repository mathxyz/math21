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
#include "../differential.h"
#include "01.h"
#include "point.h"

namespace math21 {
    namespace ad {
        Set V;
        VariableMap data(V);
        Derivative derivative(data);

        VariableMap &ad_global_get_data() {
            return data;
        }

        Derivative &ad_global_get_derivative() {
            return derivative;
        }

        void ad_clear_graph() {
            data.reset();
        }

        ad_point::ad_point() {
            id = 0;
        }

        ad_point::ad_point(NumR x) {
            *this = ad_num_const(x);
        }

        // deprecate, use to_point
        // avoid ad_point(NumN id) to disable auto cast.
        ad_point::ad_point(NumN id, NumB dummy) : id(id) {
        }

        ad_point ad_point::to_point(NumN id) {
            ad_point p;
            p.id = id;
            return p;
        }

        NumN ad_point::to_id(const ad_point &p) {
            return p.id;
        }

        ad_point::ad_point(const ad_point &p) {
            id = p.id;
        }

        ad_point::~ad_point() {}

        ad_point &ad_point::operator=(const ad_point &p) {
            id = p.id;
            return *this;
        }

        NumB ad_point::isEmpty() const {
            if (id == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        void ad_point::log(const char *name, NumN precision) const {
            log(std::cout, name, precision);
        }

        void ad_point::log(std::ostream &io, const char *name, NumN precision) const {
            ad_get_variable(*this).getValue().log(io, name, 0, 0, precision);
        }

        bool operator==(const ad_point &p1, const ad_point &p2) {
            return p1.id == p2.id;
        }

        VariableMap &ad_get_data() {
            return data;
        }

        TenR &ad_get_value(const ad_point &p) {
            MATH21_ASSERT(p.id > 0)
            return data.at(p.id).getValue();
        }

        NumN ad_get_dim_i(const ad_point &x, NumN i) {
            return ad_get_value(x).dim(i);
        }

        Variable &ad_get_variable(const ad_point &p) {
            MATH21_ASSERT(p.id > 0)
            return data.at(p.id);
        }

        NumB ad_is_const_variable(const ad_point &p) {
            if (ad_get_variable(p).getType() == variable_type_constant) {
                return 1;
            } else {
                return 0;
            }
        }

        void ad_point_set_device_type(const ad_point &p, NumN deviceType) {
            data.setDeviceType(p.id, deviceType);
        }

        NumN ad_get_device_type(const ad_point &p) {
            return ad_get_variable(p).getValue().getDeviceType();
        }

        NumB ad_point_is_cpu(const ad_point &p) {
            if (ad_get_variable(p).getValue().getDeviceType() == m21_device_type_gpu) {
                return 0;
            } else {
                return 1;
            }
        }

        ad_point ad_create_point_var(const char *name) {
            return ad_point(data.createV(name), 0);
        }

        ad_point ad_create_point_const(const char *name) {
            return ad_point(data.createC(name), 0);
        }

        ad_point ad_num_const(NumR x) {
//            m21warn("error-prone ad_num_const");
            NumN id;
            if (x == 0) {
                id = ad_global_get_constant_0();
            } else if (x == 1) {
                id = ad_global_get_constant_1();
            } else if (x == -1) {
                id = ad_global_get_constant_m1();
            } else {
                id = data.createC(math21_string_to_string(x).c_str());
                data.setValue(id, x);
            }
            return ad_point(id, 0);
        }
    }


    std::ostream &operator<<(std::ostream &io, const ad::ad_point &m) {
        m.log(io);
        return io;
    }

    void math21_io_serialize(std::ostream &out, const ad::ad_point &m, SerializeNumInterface &sn) {
        math21_io_serialize(out, m.id, sn);
    }

    void math21_io_deserialize(std::istream &in, ad::ad_point &m, DeserializeNumInterface &sn) {
        math21_io_deserialize(in, m.id, sn);
    }

    void math21_ad_destroy() {
        ad::data.clear();
    }
}