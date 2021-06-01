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
#include "sequence.h"

namespace math21 {
    // dictionary is a kind of map.
    // T is Num type. S any type.
    template<typename T, typename S>
    struct Dict {
    private:
        detail_li::Compare_index <T> comp;
        detail_li::Compare_search_index <T> comp_search;
        Seqce <T> a;
        Seqce <S> b;
        Seqce <NumN> idx;

        const Seqce <NumN> &getIdx() const {
            return idx;
        }

    private:

        void init() {
            clear();
        }

        void copyFrom(const Dict &dict) {
            dict.a.copyTo(a);
            dict.b.copyTo(b);
            dict.idx.copyTo(idx);
        }

    public:

        void serialize(std::ostream &io, SerializeNumInterface &sn) const {
            math21_io_serialize(io, a, sn);
            math21_io_serialize(io, b, sn);
            math21_io_serialize(io, idx, sn);
        }

        void deserialize(std::istream &io, DeserializeNumInterface &sn) {
            clear();
            math21_io_deserialize(io, a, sn);
            math21_io_deserialize(io, b, sn);
            math21_io_deserialize(io, idx, sn);
        }

        Dict() : comp(a), comp_search(a) {
            init();
        }

        Dict(const Dict &dict) : comp(a), comp_search(a) {
            init();
            copyFrom(dict);
        }

        Dict &operator=(const Dict &B) {
            MATH21_ASSERT_NOT_CALL(0)
            return *this;
        }

        virtual ~Dict() {
            clear();
        }

        NumN size() const {
            return a.size();
        }

        NumB isEmpty() const {
            if (size() == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        const Seqce <T> &getX() const {
            MATH21_ASSERT_NOT_CALL(0)
            return a;
        }

        const Seqce <S> &getY() const {
            MATH21_ASSERT_NOT_CALL(0)
            return b;
        }

        template<template<typename> class Container>
        void getX(Container<T> &X) const {
            X.setSize(size());
            for (NumN i = 1; i <= size(); ++i) {
                X(i) = keyAtIndex(i);
            }
        }

        template<template<typename> class Container>
        void getY(Container<S> &Y) const {
            Y.setSize(size());
            for (NumN i = 1; i <= size(); ++i) {
                Y(i) = valueAtIndex(i);
            }
        }

        const S &valueAt(const T &x) const {
            NumN id = has(x);
            MATH21_ASSERT(id)
            return valueAtIndex(id);
        }

        S &valueAt(const T &x) {
            NumN id = has(x);
            MATH21_ASSERT(id, "no key " << x << " in dict")
            return valueAtIndex(id);
        }

        // get key at position i.
        // position is virtual.
        const T &keyAtIndex(NumN i) const {
            return a(getIdx()(i));
        }

        const S &valueAtIndex(NumN i) const {
            return b(getIdx()(i));
        }

        S &valueAtIndex(NumN i) {
            return b(getIdx()(i));
        }

        // 0: not find
        NumN has(const T &x) const {
            NumN id;
            id = detail_li::binary_search(idx, x, idx.size(), comp_search);
            return id;
        }

        void add(const T &x, const S &y) {
            a.push(x);
            b.push(y);
            idx.push(a.size());
            detail_li::insertion_sort_insert(idx, idx.size(), comp);
        }

        template<template<typename> class Container>
        void add(const Container<T> &X, const S &y) {
            for (NumN i = 1; i <= X.size(); ++i) {
                add(X(i), y);
            }
        }

        void clear() {
            a.clear();
            b.clear();
            idx.clear();
        }

        void log(const char *s = 0) const {
            log(std::cout, s);
        }

        // todo: merge log of different map class using template.
        void log(std::ostream &io, const char *s = 0) const {
            if (s == 0) {
                s = "";
            }
            io << "Dict " << s << ":\n";
            io << "size: " << size() << "\n";
            for (NumN i = 1; i <= size(); ++i) {
                NumN id = getIdx()(i);
                io << "(" << a(id) << ", " << b(id) << ")\n";
            }
        }
    };

    template<typename T, typename S>
    void math21_io_serialize(std::ostream &io, const Dict<T, S> &m, SerializeNumInterface &sn) {
        m.serialize(io, sn);
    }

    template<typename T, typename S>
    void math21_io_deserialize(std::istream &io, Dict<T, S> &m, DeserializeNumInterface &sn) {
        m.deserialize(io, sn);
    }
}