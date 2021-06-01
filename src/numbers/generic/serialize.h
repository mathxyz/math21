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
    namespace detail {

        template<typename T, template<typename> class Container>
        void math21_io_serialize_container(std::ostream &out, const Container<T> &m, SerializeNumInterface &sn) {
            NumN i;
            NumN n = m.size();
            math21_io_serialize(out, n, sn);
            for (i = 1; i <= n; i++) {
                math21_io_serialize(out, m(i), sn);
            }
        }

        template<typename T, template<typename> class Container>
        void math21_io_deserialize_container(std::istream &in, Container<T> &m, DeserializeNumInterface &sn) {
            NumN i;
            NumN n;
            math21_io_deserialize(in, n, sn);
            if (m.size() != n) {
                m.setSize(n);
            }
            for (i = 1; i <= n; i++) {
                math21_io_deserialize(in, m.at(i), sn);
            }
        }
    }
}