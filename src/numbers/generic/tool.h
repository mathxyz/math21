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

// return 0 if not find, 1 success. index will be first index of x.
    template<typename VecType>
    NumB
    math21_operator_container_print(const VecType &v, std::ostream &io, const char *name = 0, const char *gap = 0,
                                    NumN precision = 3) {
        if (name) {
            io << "LightVector " << name << ", size = " << v.size() << "\n";
        }
        io << std::setprecision(precision);
        if (!gap) {
            gap = " ";
        }

        using namespace std;
        const streamsize old = io.width();

        // first figure out how wide we should make each field
        string::size_type w = 0;
        ostringstream sout;
        for (NumN r = 1; r <= v.size(); ++r) {
            sout << v(r);
            w = std::max(sout.str().size(), w);
            sout.str("");
        }


        // now actually print it
        for (NumN r = 1; r <= v.size(); ++r) {
            io.width(static_cast<streamsize>(w));
            io << v(r);
            io << gap;
        }
        io << "\n";

        io.width(old);
        return 1;
    }

    // return 0 if not find, 1 success. index will be first index of x.
    template<typename T, template<typename> class Container, typename T2>
    NumB math21_operator_container_find_first(const Container<T> &v, const T2 &x, NumN &index) {
        for (NumN i = 1; i <= v.size(); ++i) {
            if (v(i) == x) {
                index = i;
                return 1;
            }
        }
        return 0;
    }

    // return 0 if not find, 1 success. index will be first index of x.
    template<typename T, template<typename> class Container, typename T2, typename Compare>
    NumB math21_operator_container_find_first_using_compare(const Container<T> &v,
                                                            const T2 &x, NumN &index, Compare &f) {
        for (NumN i = 1; i <= v.size(); ++i) {
            if (f.compare(v(i), x) == 0) {
                index = i;
                return 1;
            }
        }
        return 0;
    }

    // return 0 if not find, 1 success. index will be first index of x.
    template<typename T, template<typename> class Container, typename T2>
    NumB math21_operator_container_find_last(const Container<T> &v, const T2 &x, NumN &index) {
        for (NumN i = v.size(); i >= 1; --i) {
            if (v(i) == x) {
                index = i;
                return 1;
            }
        }
        return 0;
    }
}