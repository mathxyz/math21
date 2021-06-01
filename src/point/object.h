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
#include "point_cc.h"

namespace math21 {
    class m21object {
    private:
        m21point point;
    public:
        m21object();

        ~m21object();

        m21object(const m21object &o);

        m21object &operator=(const m21object &o);

        m21object &create(NumN type);

        template<typename T>
        m21object(const T &x) {
            point = math21_cast_to_point(x);
        }

        void clear();

        NumB isEmpty() const;

        template<typename T>
        T &get() {
            return math21_cast_to_T<T>(point);
        }

        template<typename T>
        const T &get() const {
            return math21_cast_to_T<T>(point);
        }

        NumB isTenN() const;

        NumB isTenN8() const;

        NumB isTenZ() const;

        NumB isTenR() const;

        NumB log(const char *name = 0) const;

        NumB log(std::ostream &io, const char *name = 0) const;
    };

    std::ostream &operator<<(std::ostream &out, const m21object &m);
}