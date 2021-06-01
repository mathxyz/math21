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

    template<typename T, typename VecType>
    struct literal_assign_helper_container {
        /*
            This struct is a helper struct returned by the operator<<() function below.  It is
            used primarily to enable us to put MATH21_CASSERT statements on the usage of the
            operator<< form of vector assignment.
        */

        literal_assign_helper_container(const literal_assign_helper_container &item) : m(item.m), n(item.n),
                                                                                       has_been_used(0) {
        }

        explicit literal_assign_helper_container(VecType *m_) : m(m_), n(1), has_been_used(0) {
            next();
        }

        void clear() {
            MATH21_ASSERT(!has_been_used || n - 1 == m->size(),
                          "You have used the vector comma based assignment incorrectly by failing to\n"
                          "supply a full set of values for every element of a vector object.\n");
        }

        ~literal_assign_helper_container() {
            clear();
        }

        const literal_assign_helper_container &operator,(const T &val) const {
            MATH21_ASSERT(n <= m->size(),
                          "You have used the vector comma based assignment incorrectly by attempting to\n" <<
                                                                                                           "supply more values than there are elements in the vector object being assigned to.\n\n"
                                                                                                           <<
                                                                                                           "Did you forget to call setSize()?"
                                                                                                           << "\n\t n: "
                                                                                                           << n
                                                                                                           << "\n\t m->size(): "
                                                                                                           << m->size());
            (*m).at(n) = val;
            next();
            has_been_used = 1;
            return *this;
        }

    private:

        void next(
        ) const {
            ++n;
        }

        VecType *m;
        mutable NumN n;
        mutable NumB has_been_used;
    public:
        VecType *getContainer() const {
            return m;
        }
    };
}