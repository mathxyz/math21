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

#include "point_c.h"
#include "object.h"

namespace math21 {
    m21object::m21object() {
        point = math21_point_init(point);
    }

    m21object::~m21object() {
        clear();
    }

    m21object::m21object(const m21object &o) {
        point = math21_point_share_assign(o.point);
    }

    m21object &m21object::operator=(const m21object &o) {
        if (this != &o) {
            if (!isEmpty())clear();
            point = math21_point_share_assign(o.point);
        }
        return *this;
    }

    m21object &m21object::create(NumN type) {
        if (!isEmpty())clear();
        point = math21_point_create_by_type(type);
        return *this;
    }

    void m21object::clear() {
        point = math21_point_destroy(point);
    }

    NumB m21object::isEmpty() const {
        return math21_point_is_empty(point);
    }

    NumB m21object::isTenN() const {
        if (point.type == m21_type_TenN) {
            return 1;
        }
        return 0;
    }

    NumB m21object::isTenN8() const {
        if (point.type == m21_type_TenN8) {
            return 1;
        }
        return 0;
    }

    NumB m21object::isTenZ() const {
        if (point.type == m21_type_TenZ) {
            return 1;
        }
        return 0;
    }

    NumB m21object::isTenR() const {
        if (point.type == m21_type_TenR) {
            return 1;
        }
        return 0;
    }

    NumB m21object::log(const char *name) const {
        math21_point_log_cc(point, name);
        return 1;
    }

    NumB m21object::log(std::ostream &io, const char *name) const {
        math21_point_log_cc(io, point, name);
        return 1;
    }

    std::ostream &operator<<(std::ostream &out, const m21object &m) {
        m.log(out);
        return out;
    }
}