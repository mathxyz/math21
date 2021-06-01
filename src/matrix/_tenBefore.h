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
#include "array.h"

namespace math21 {
    template<typename T>
    class Tensor;

    namespace detail {

        // Todo: will deprecate, and use literal_assign_helper_container.
        template<class T>
        struct literal_assign_helper_tensor {
            /*
                This struct is a helper struct returned by the operator=() function below.  It is
                used primarily to enable us to put MATH21_CASSERT statements on the usage of the
                operator= form of matrix assignment.
            */

            literal_assign_helper_tensor(const literal_assign_helper_tensor &item);

            explicit literal_assign_helper_tensor(Tensor<T> &m_);

            virtual ~literal_assign_helper_tensor();

            void clear();

            const literal_assign_helper_tensor &operator,(
                    const T &val
            ) const;

        private:

            friend class Tensor<T>;

            Tensor<T> &A;
            mutable ArrayN d;
            mutable NumB isFull;
            mutable NumB has_been_used;

            void next() const;

        public:
            Tensor<T> &getTensor() const {
                return A;
            }
        };
    }
}