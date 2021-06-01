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

#include <iostream>
#include <cstdio>
//#include <stdexcept>
#include <vector>
#include "inner.h"

/*
 * A 1d array with supporting push and pop.
 * See Array class.
 * */
namespace math21 {
    // seqce
    template<typename T>
    class Seqce {
    private:
        std::vector<T> v;

        void init() {
            v.clear();
        }

    public:
        Seqce() {
            init();
        }

        Seqce(NumN n) {
            init();
            math21_tool_std_vector_resize(v, n);
        }

        // Copy constructor
        Seqce(const Seqce &B) {
//            MATH21_ASSERT(0, "Copy constructor.");
            B.copyTo(*this);
        }

        // Copy constructor
        template<typename S>
        Seqce(const Seqce<S> &B) {
            MATH21_ASSERT(0, "Copy constructor.");
//            B.copyTo(*this);
        }

        virtual ~Seqce() {
            clear();
        }

        NumB isEqual(const Seqce &B) const {
            return math21_operator_container_isEqual(*this, B, 0);
        }

        NumB isEmpty() const {
            if (v.empty()) {
                return 1;
            }
            return 0;
        }

        NumN size() const {
            if (v.empty()) {
                return 0;
            }
            return (NumN)v.size();
        }

        // seems not used.
        //from left to right.
        NumN shape() const {
            return size();
        }

        // seems not used.
        // deprecate
        NumN nrows() const {
            return size();
        }

        // seems not used.
        NumN ncols() const {
            if (v.empty()) {
                return 0;
            }
            return 1;
        }

        // data may not be zero
        void setSize(NumN n) {
            if (n == 0) {
                clear();
                return;
            }
            if (n == size()) {
                return;
            }
            math21_tool_std_vector_resize(v, n);
        }

        const T &operator()(NumN n, NumN m) const {
            MATH21_ASSERT_INDEX(n >0);
            MATH21_ASSERT_INDEX(n <= size() && m == 1);
            return v[n - 1];
        }

        const T &operator()(NumN n) const {
            MATH21_ASSERT_INDEX(n >0);
            MATH21_ASSERT_INDEX(n <= size());
            return v[n - 1];
        }

        T &operator()(NumN n, NumN m) {
            MATH21_ASSERT_INDEX(n >0);
            MATH21_ASSERT_INDEX(n <= size() && m == 1);
            return v[n - 1];
        }

        T &operator()(NumN n) {
            MATH21_ASSERT_INDEX(n >0);
            MATH21_ASSERT_INDEX(n <= size());
            return v[n - 1];
        }

        T &at(NumN n) {
            MATH21_ASSERT_INDEX(n >0);
            MATH21_ASSERT_INDEX(n <= size());
            return v[n - 1];
        }

        // push back
        void push(const T &x) {
            v.push_back(x);
        }

        // push back
        template<template<typename> class Container>
        void push(const Container<T> &xs) {
            for (NumN i = 1; i <= xs.size(); ++i) {
                v.push_back(xs(i));
            }
        }

        T &getLast() {
            return at(v.size());
        }

        void removeLast() {
            v.pop_back();
        }

        void clear() {
            v.clear();
        }

        void logInfo(const char *s = 0) const {
            logInfo(std::cout, s);
        }

        void logInfo(std::ostream &io, const char *s = 0) const {
            if (s) {
                io << "Seqce " << s << ":\n";
            }
            io << "dims: " << size() << "\n";
        }

        NumB log(const char *s = 0, NumB row_print = 0, NumB isUsingComma = 1) const {
            return log(std::cout, s, row_print, isUsingComma);
        }

        NumB log(std::ostream &io, const char *s = 0, NumB row_print = 0, NumB isUsingComma = 1) const {
            if (s) {
                io << "Seqce " << s << ":\n";
            }
            if (!row_print) {
                for (NumN i = 1; i <= size(); ++i) {
                    const T &x = (*this)(i);
                    math21_print(io, x, math21_string_to_string(i).c_str());
                }
            } else {
                if (size() == 0) {
                    io << "{}\n";
                    return 1;
                }

                for (NumN i = 1; i <= size(); ++i) {
                    if (i == 1) {
                        io << "{";
                    }
                    if (i != size()) {
                        io << (*this)(i);
                        if (isUsingComma) {
                            io << ", ";
                        } else {
                            io << " ";
                        }
                    } else {
                        io << (*this)(i) << "}\n";
                    }
                }
            }
            return 1;
        }

        NumB isContinuous() {
            return 0;
        }

        void copyTo(Seqce<T> &B) const {
            if (size() == 0) {
                B.clear();
                return;
            }
            if (B.size() != size()) {
                B.setSize(size());
            }
            B.assign(*this);
        }

        void copyFrom(const Seqce<T> &B) {
            B.copyTo(*this);
        }

        // not test
        void swap(Seqce<T> &B) {
            v.swap(B.v);
        }

        //assignment
        Seqce<T> &operator=(const Seqce<T> &B) {
//            MATH21_ASSERT(0, "=");
            assign(B);
            return *this;
        }

        //assignment
        template<typename S>
        Seqce<T> &operator=(const Seqce<S> &B) {
            MATH21_ASSERT(0, "=");
//            assign(B);

//            return *this;
        }

        void assign(const Seqce<T> &B) {
            MATH21_ASSERT(B.size() == size(),
                          "vector size doesn't match in assign");
            if (this != &B) {
                NumN i;
                for (i = 1; i <= size(); i++) (*this).at(i) = B(i);
            }
        }

        void assign(const T &a) {
            NumN i;
            for (i = 1; i <= size(); i++) (*this).at(i) = a;
        }

        template<typename S>
        Seqce &operator=(
                const literal_assign_helper_container<S, Seqce> &val
        ) {
            MATH21_ASSERT(0, "literal_assign_helper_container")
//            val.m->copyTo(*this);
//            return *this;
        }

        template<typename S>
        const literal_assign_helper_container<T, Seqce> operator=(
                const S &val
        ) {
            // assign the given value to every spot in this matrix
            assign(val);
            // Now return the literal_assign_helper_container so that the user
            // can use the overloaded comma notation to initialize
            // the matrix if they want to.
            return literal_assign_helper_container<T, Seqce>(this);
        }

        void sort();

        void letters() {
            NumN i;
            NumN n = size();
            for (i = 1; i <= n; ++i) {
                at(i) = i;
            }
        }
    };

    template<typename T>
    std::ostream &operator<<(std::ostream &io, const Seqce<T> &m) {
        m.log(io);
        return io;
    }

    template<typename T>
    NumN math21_type_get(const Seqce<T> &m) {
        return m21_type_Seqce;
    }

    template<typename T>
    void math21_io_serialize(std::ostream &out, const Seqce<T> &m, SerializeNumInterface &sn) {
        detail::math21_io_serialize_container(out, m, sn);
    }

    template<typename T>
    void math21_io_deserialize(std::istream &in, Seqce<T> &m, DeserializeNumInterface &sn) {
        detail::math21_io_deserialize_container(in, m, sn);
    }

}