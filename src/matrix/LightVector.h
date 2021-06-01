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

    // LightVector is a wrapper of c array.
    // So we can easily access element using operator() with index starting from 1 instead of 0.
    //
    template<typename T>
    class LightVector {
    private:
        NumN n;
        T *data; // data may be constant if isDataExternal = 1.
        NumB isDataExternal;

        void init() {
            data = 0;
            n = 0;
            isDataExternal = 0;
        }

    public:
        LightVector() {
            init();
        }

        LightVector(const LightVector &vector) {
            init();
            copyFrom(vector);
        }

        LightVector(NumN n, T *data) {
            this->n = n;
            this->data = data - 1;
            isDataExternal = 1;
        }

        LightVector(NumN n, const T *data) {
            this->n = n;
            this->data = (T *) data - 1;
            isDataExternal = 1;
        }

        explicit LightVector(NumN n) {
            this->n = n;
            data = (T *) math21_vector_create_buffer_cpu(n, sizeof(T));
            isDataExternal = 0;
        }

        void setSize(NumN n0) {
            MATH21_ASSERT(!isDataExternal)
            if (n0 != n) {
                data = (T *) math21_vector_setSize_buffer_cpu(data, n, sizeof(T));
                n = n0;
            }
        }

        void copyFrom(const LightVector &v) {
            setSize(v.size());
            math21_vector_copy_buffer_cpu(data, v.data, n, sizeof(T));
        }

        void clear() {
            if (!isDataExternal) {
                math21_vector_free_cpu(data);
                n = 0;
            }
        }

        virtual ~LightVector() {
            clear();
        }

        // j1 >= 1
        virtual T &at(NumN j1) {
            return data[j1];
        }

        // j1 >= 1
        virtual T &operator()(NumN j1) {
            return data[j1];
        }

        // j1 >= 1
        virtual const T &operator()(NumN j1) const {
            return data[j1];
        }

        NumB isEmpty() const {
            return n == 0 ? (NumB) 1 : (NumB) 0;
        }

        NumN size() const {
            if (isEmpty()) {
                return 0;
            }
            return n;
        }

        NumB log(const char *name = 0, const char *gap = 0, NumN precision = 3) const {
            return math21_operator_container_print(*this, std::cout, name, gap, precision);
        }
    };
}