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
    /*
     * y=f(x)
     *
     * */

    // Default: 0-based indexing
    // T is NumN, NumZ, or NumR.
    // this function is just a wrapper around tensor,
    // and can be viewed as an index-generalized tensor.
    template<typename T>
    class IndexFunctional : public think::Operator {
    private:
        VecZ neg_index_offset; // index input - neg index offset = tensor index
        Tensor <T> v;

        template<typename VecZVecNType>
        T &valueAt(const VecZVecNType &index) {
            if (index.size() == 1) {
                return operator()(index(1));
            } else if (index.size() == 2) {
                return operator()(index(1), index(2));
            } else if (index.size() == 3) {
                return operator()(index(1), index(2), index(3));
            }
            MATH21_ASSERT(math21_operator_container_is_larger(index, neg_index_offset),
                          "" << index.log("index") << "\n"
                             << neg_index_offset.log("neg_index_offset") << "\n")
            VecN index_cur(index.size());
            math21_operator_container_subtract_to_C(index, neg_index_offset, index_cur);
            return v.operator()(index_cur);
        }

        template<typename VecZVecNType>
        const T &valueAt(const VecZVecNType &index) const {
            if (index.size() == 1) {
                return operator()(index(1));
            } else if (index.size() == 2) {
                return operator()(index(1), index(2));
            } else if (index.size() == 3) {
                return operator()(index(1), index(2), index(3));
            }
            MATH21_ASSERT(math21_operator_container_is_larger(index, neg_index_offset),
                          "" << index.log("index") << "\n"
                             << neg_index_offset.log("neg_index_offset") << "\n")

            VecN index_cur(index.size());
            math21_operator_container_subtract_to_C(index, neg_index_offset, index_cur);
            return v(index_cur);
        }

    public:
        IndexFunctional() {}

        //must be virtual.
        virtual ~IndexFunctional() {}

        IndexFunctional(const IndexFunctional<T> &f) {
            *this = f;
        }

        IndexFunctional<T> &operator=(const IndexFunctional<T> &f) {
            neg_index_offset.copyFrom(f.neg_index_offset);
            v.copyFrom(f.v);
            return *this;
        }

        //from left to right, d1 is lowest.
        IndexFunctional(NumN d1) {
            setSize(d1);
        }

        //from left to right, d2 is lowest.
        IndexFunctional(NumN d1, NumN d2) {
            setSize(d1, d2);
        }

        //from left to right, d3 is lowest.
        IndexFunctional(NumN d1, NumN d2, NumN d3) {
            setSize(d1, d2, d3);
        }

        void setStartIndex(const VecZ &start) {
            if (!v.isEmpty()) {
                MATH21_ASSERT(v.dims() == start.size())
            }
            neg_index_offset.setSize(start.size());
            math21_operator_container_subtract_A_k(start, 1, neg_index_offset);
        }

        void setSize(const VecN &d) {
            if (neg_index_offset.isEmpty() || neg_index_offset.size() != d.size()) {
                VecZ start(d.size());
                start = 0;
                setStartIndex(start);
            }
            v.setSize(d);
        }

        //from left to right, d1 is lowest.
        void setSize(NumN d1) {
            VecN d(1);
            d = d1;
            setSize(d);
        }

        //from left to right, d2 is lowest.
        void setSize(NumN d1, NumN d2) {
            VecN d(2);
            d = d1, d2;
            setSize(d);
        }

        //from left to right, d3 is lowest.
        void setSize(NumN d1, NumN d2, NumN d3) {
            VecN d(3);
            d = d1, d2, d3;
            setSize(d);
        }

        //from left to right, d3 is lowest.
        void setSize(NumN d1, NumN d2, NumN d3, NumN d4) {
            VecN d(4);
            d = d1, d2, d3, d4;
            setSize(d);
        }

        //from left to right, d1 is lowest.
        void setStartIndex(NumZ d1) {
            VecZ d(1);
            d = d1;
            setStartIndex(d);
        }

        //from left to right, d2 is lowest.
        void setStartIndex(NumZ d1, NumZ d2) {
            VecZ d(2);
            d = d1, d2;
            setStartIndex(d);
        }

        //from left to right, d3 is lowest.
        void setStartIndex(NumZ d1, NumZ d2, NumZ d3) {
            VecZ d(3);
            d = d1, d2, d3;
            setStartIndex(d);
        }

        //from left to right, d4 is lowest.
        void setStartIndex(NumZ d1, NumZ d2, NumZ d3, NumZ d4) {
            VecZ d(4);
            d = d1, d2, d3, d4;
            setStartIndex(d);
        }

        NumB isEmpty() const {
            return v.isEmpty();
        }

        NumN nrows() const {
            return v.nrows();
        }

        NumN ncols() const {
            return v.ncols();
        }

        T &operator()(const VecZ &index) {
            return valueAt(index);
        }

        T &operator()(const VecN &index) {
            return valueAt(index);
        }

        const T &operator()(const VecZ &index) const {
            return valueAt(index);
        }

        const T &operator()(const VecN &index) const {
            return valueAt(index);
        }

        template<template<typename> class Container, typename NumZType>
        T &operator()(const Container<NumZType> &index_a, NumZType j1) {
            VecZ index(index_a.size() + 1);
            math21_operator_container_set_partially(index_a, index, 0, 0, index_a.size());
            index(index.size()) = j1;
            return this->operator()(index);
        }

        T &operator()(NumZ j1) {
            j1 = j1 - neg_index_offset(1);
            MATH21_ASSERT(j1 > 0)
            return v(j1);
        }

        const T &operator()(NumZ j1) const {
            j1 = j1 - neg_index_offset(1);
            MATH21_ASSERT(j1 > 0)
            return v(j1);
        }

        T &operator()(NumZ j1, NumZ j2) {
            NumZ _j1 = j1 - neg_index_offset(1);
            NumZ _j2 = j2 - neg_index_offset(2);
            MATH21_ASSERT(_j1 > 0 && _j2 > 0, "j1 = " << j1 << ", j2 = " << j2
                                                      << "\n" << neg_index_offset.log(__FUNCTION__)
                                                      << "\n"
            );
            return v(_j1, _j2);
        }

        const T &operator()(NumZ j1, NumZ j2) const {
            j1 = j1 - neg_index_offset(1);
            j2 = j2 - neg_index_offset(2);
            MATH21_ASSERT(j1 > 0 && j2 > 0)
            return v(j1, j2);
        }

        T &operator()(NumZ j1, NumZ j2, NumZ j3) {
            j1 = j1 - neg_index_offset(1);
            j2 = j2 - neg_index_offset(2);
            j3 = j3 - neg_index_offset(3);
            MATH21_ASSERT(j1 > 0 && j2 > 0 && j3 > 0)
            return v(j1, j2, j3);
        }

        const T &operator()(NumZ j1, NumZ j2, NumZ j3) const {
            j1 = j1 - neg_index_offset(1);
            j2 = j2 - neg_index_offset(2);
            j3 = j3 - neg_index_offset(3);
            MATH21_ASSERT(j1 > 0 && j2 > 0 && j3 > 0)
            return v(j1, j2, j3);
        }

        NumN size() const {
            return v.size();
        }

        void setTensor(const Tensor <T> &m) {
            VecN d;
            setSize(m.shape(d));
            v.copyFrom(m);
        }

        // Todo: make it private or not
        Tensor <T> &getTensor() {
            return v;
        }

        const Tensor <T> &getTensor() const {
            return v;
        }

        NumZ getTensor_index_offset(NumN i) const {
            return -neg_index_offset(i);
        }

        template<typename VecZType>
        void get_start_index(VecZType &start) const {
            MATH21_ASSERT(!isEmpty())
            if (start.isSameSize(neg_index_offset.size()) == 0) {
                start.setSize(neg_index_offset.size());
            }
            math21_operator_container_add_A_k(neg_index_offset, 1, start);
        }

        template<typename VecZType>
        void get_end_index(VecZType &end) const {
            MATH21_ASSERT(!isEmpty())
            if (end.isSameSize(neg_index_offset.size()) == 0) {
                end.setSize(neg_index_offset.size());
            }
            // addition between NumN and NumZ, so end must has dtype NumZ
            math21_operator_container_addToC(neg_index_offset, v.shape(), end);
        }

        template<typename VecZType>
        void getIndex(NumN i, VecZType &index) const {
            VecZType start(neg_index_offset.size());
            get_start_index(start);
            math21_operator_number_index_1d_to_nd(index, i, v.shape(), start);
        }

        void log(const char *name = 0, NumB isLogDetail = 0) const {
            if (name == 0) {
                name = "";
            }
            if (isLogDetail) {
                neg_index_offset.log(math21_string_concatenate(name, " neg_index_offset").c_str());
            }
            v.log(name);
        }
    };
}
