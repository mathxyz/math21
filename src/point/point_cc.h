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
    T &math21_cast_to_T(m21point point) {
        if (math21_type_get<T>() != point.type) {
            MATH21_ASSERT(0,
                          "type mismatch, type = " << math21_type_name(point.type) << ", math21_type_get<T>() = "
                                                   << math21_type_get<T>()
                                                   << ", math21_type_name<T>() = " << math21_type_name<T>()
                                                   << "\n");
        }
        MATH21_ASSERT(point.p, "Null pointer in cast")
        return *(T *) point.p;
    }

    template<>
    m21point &math21_cast_to_T(m21point point);

    template<typename T>
    m21point math21_cast_to_point(const T &x) {
        m21point point = {0};
        point.type = math21_type_get<T>();
        point.p = (void *) (&x);
        return point;
    }

    // no conversion
    template<>
    m21point math21_cast_to_point(const m21point &x);

    void math21_point_log_cc(m21point point, const char *s = 0);

    void math21_point_log_cc(std::ostream &out, m21point point, const char *s = 0);

    void math21_io_serialize(std::ostream &out, const m21point &point, SerializeNumInterface &sn);

    void math21_io_deserialize(std::istream &in, m21point &point, DeserializeNumInterface &sn);

    template<typename T>
    NumB math21_io_load(const char *path, T &x) {
        m21point point = math21_point_load(path);
        if(math21_point_is_empty(point)){
            return 0; // no need to destroy
        }
        x = math21_cast_to_T<T>(point);
        math21_point_destroy(point);
        return 1;
    }

    template<typename T>
    NumB math21_io_save(const char *path, const T &x) {
        return math21_point_save(math21_cast_to_point(x), path);
    }
}