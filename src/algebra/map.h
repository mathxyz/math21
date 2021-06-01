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
#include <map>

namespace math21 {

    template<typename T, typename S>
    struct _Map {
    private:
        S b_dummy;
        std::map<T, S> data;

        void init();

    public:

        _Map();

        virtual ~_Map();

        NumN size() const;

        NumB isEmpty() const;

//        const Seqce <T> &getX() const;

        std::map<T, S> &getData();

        const std::map<T, S> &getData() const;

        void getX(_Set <T> &keys) const;

        void getY(Seqce <S> &vs) const;

//        const Seqce <S> &getY() const;

        NumB get(const T &x, S &y) const;

        S &valueAt(const T &x);

        S &valueAt(const T &x, NumB &flag);

        const S &valueAt(const T &x) const;

        const S &valueAt(const T &x, NumB &flag) const;

        // get key at position i.
        // position is virtual.
//        const T &keyAtIndex(NumN i) const;

//        const S &valueAtIndex(NumN i) const;

        NumN has(const T &x) const;

        NumB remove(const T &x);

        void add(const T &x, const S &y);

        void add(const _Set <T> &X, const S &y);

        void add(const Relation <T, S> &r);

        void clear();

        void restrictTo(const _Set <T> &X, _Map<T, S> &dst);

        void log(const char *s = 0) const;

        void log(std::ostream &io, const char *s = 0) const;
    };

    template<typename T, typename S>
    void _Map<T, S>::init() {
        clear();
    }

    template<typename T, typename S>
    _Map<T, S>::_Map() {
        init();
    }

    template<typename T, typename S>
    _Map<T, S>::~_Map() {
        clear();
    }

    template<typename T, typename S>
    NumN _Map<T, S>::size() const {
        return (NumN) data.size();
    }

    template<typename T, typename S>
    NumB _Map<T, S>::isEmpty() const {
        if (size() == 0) {
            return 1;
        } else {
            return 0;
        }
    }

//    template<typename T, typename S>
//    const Seqce <T> &_Map<T, S>::getX() const {
//        math21_tool_assert(0);
//    }

    template<typename T, typename S>
    std::map<T, S> &_Map<T, S>::getData() {
        return data;
    }

    template<typename T, typename S>
    const std::map<T, S> &_Map<T, S>::getData() const {
        return data;
    }

    template<typename T, typename S>
    void _Map<T, S>::getX(_Set <T> &keys) const {
        keys.clear();
        if (isEmpty()) {
            return;
        }
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            keys.add(itr->first);
        }
    }

    template<typename T, typename S>
    void _Map<T, S>::getY(Seqce <S> &vs) const {
        vs.clear();
        if (isEmpty()) {
            return;
        }
        vs.setSize(size());
        NumN i = 1;
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            vs.at(i) = itr->second;
            ++i;
        }
    }

//    template<typename T, typename S>
//    const Seqce <S> &_Map<T, S>::getY() const {
//        math21_tool_assert(0);
//    }

    template<typename T, typename S>
    NumB _Map<T, S>::get(const T &x, S &y) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            y = itr->second;
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T, typename S>
    S &_Map<T, S>::valueAt(const T &x) {
        auto itr = data.find(x);
        if (itr == data.end()) {
            MATH21_ASSERT(0);
        }
        return itr->second;
    }

    template<typename T, typename S>
    S &_Map<T, S>::valueAt(const T &x, NumB &flag) {
        auto itr = data.find(x);
        if (itr != data.end()) {
            flag = 1;
            return itr->second;
        }
        flag = 0;
        return b_dummy;
    }

    template<typename T, typename S>
    const S &_Map<T, S>::valueAt(const T &x) const {
//        for (NumN i = 1; i <= a.size(); ++i) {
//            if (a(i) == x) {
//                return b(i);
//            }
//        }
//        MATH21_ASSERT(0)

        auto itr = data.find(x);
        if (itr != data.end()) {
            return itr->second;
        }
        MATH21_ASSERT(0)
        return b_dummy;
    }

    template<typename T, typename S>
    const S &_Map<T, S>::valueAt(const T &x, NumB &flag) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            flag = 1;
            return itr->second;
        }
        flag = 0;
        return b_dummy;

//        for (NumN i = 1; i <= a.size(); ++i) {
//            if (a(i) == x) {
//                flag = 1;
//                return b(i);
//            }
//        }
//        flag = 0;
//        return b_dummy;
    }

    // get key at position i.
    // position is virtual.
//    template<typename T, typename S>
//    const T &_Map<T, S>::keyAtIndex(NumN i) const {
//        math21_tool_assert(0);
//        return a(i);
//    }

//    template<typename T, typename S>
//    const S &_Map<T, S>::valueAtIndex(NumN i) const {
//        math21_tool_assert(0);
//        return b(i);
//    }


    template<typename T, typename S>
    NumN _Map<T, S>::has(const T &x) const {
        auto itr = data.find(x);
        if (itr != data.end()) {
            return 1;
        }
        return 0;
//        for (NumN i = 1; i <= a.size(); ++i) {
//            if (a(i) == x) {
//                return i;
//            }
//        }
//        return 0;
    }

    template<typename T, typename S>
    NumB _Map<T, S>::remove(const T &x) {
        if (data.erase(x) == 1) {
            return 1;
        } else {
            return 0;
        }
    }

    // It will fail if (x, *) exists
    template<typename T, typename S>
    void _Map<T, S>::add(const T &x, const S &y) {
        NumB flag = (NumB) data.insert(std::pair<T, S>(x, y)).second;
        if (!flag) {
            m21warn("add failed!");
            std::cout << "x = " << x << ", y = " << y << std::endl;
        }
    }

    template<typename T, typename S>
    void _Map<T, S>::add(const _Set <T> &X, const S &y) {
        for (NumN i = 1; i <= X.size(); ++i) {
            add(X(i), y);
        }
    }

    template<typename T, typename S>
    void _Map<T, S>::add(const Relation <T, S> &r) {
        for (NumN i = 1; i <= r.size(); ++i) {
            add(r.keyAtIndex(i), r.valueAtIndex(i));
        }
    }

    template<typename T, typename S>
    void _Map<T, S>::clear() {
        data.clear();
    }

    // restrict f: X -> Y to X0 to get f: Xs -> Y, with Xs = intersect(X, X0)
    template<typename T, typename S>
    void _Map<T, S>::restrictTo(const _Set <T> &X0, _Map<T, S> &dst) {
        dst.clear();
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            if (X0.contains(itr->first)) {
                dst.add(itr->first, itr->second);
            }
        }
    }

    template<typename T, typename S>
    void _Map<T, S>::log(const char *s) const {
        log(std::cout, s);
    }

    template<typename T, typename S>
    void _Map<T, S>::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = "";
        }
        io << "_Map " << s << ":\n";
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            io << "(" << itr->first << ", " << itr->second << ")\n";
        }
//        for (NumN i = 1; i <= size(); ++i) {
//            io << "(" << a(i) << ", " << b(i) << ")\n";
//        }
    }
}