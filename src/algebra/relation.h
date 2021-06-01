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

    // It use insertion order, not sorted order.
    // just relation, but not check duplicate pairs.
    // not equivalence relation
    template<typename T, typename S>
    struct Relation {
    private:
        Seqce<T> X;
        Seqce<S> Y;

        void init();

    public:

        Relation();

        virtual ~Relation();

        NumN size() const;

        NumB isEmpty() const;

        Seqce<T> &getX() { return X; }

        const Seqce<T> &getX() const { return X; }

        Seqce<S> &getY() { return Y; }

        const Seqce<S> &getY() const { return Y; }

        // get key at position i.
        // position is virtual.
        const T &keyAtIndex(NumN i) const;

        const S &valueAtIndex(NumN i) const;

        NumN has(const T &x) const;

        void add(const T &x, const S &y);

        void add(const Seqce<T> &X, const S &y);

        void clear();

        void log(const char *s = 0) const;

        void log(std::ostream &io, const char *s = 0) const;
    };

    template<typename T, typename S>
    void Relation<T, S>::init() {
        clear();
    }

    template<typename T, typename S>
    Relation<T, S>::Relation() {
        init();
    }

    template<typename T, typename S>
    Relation<T, S>::~Relation() {
        clear();
    }

    template<typename T, typename S>
    NumN Relation<T, S>::size() const {
        return (NumN) X.size();
    }

    template<typename T, typename S>
    NumB Relation<T, S>::isEmpty() const {
        if (size() == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    // get key at position i.
    // position is virtual.
    template<typename T, typename S>
    const T &Relation<T, S>::keyAtIndex(NumN i) const {
        return X(i);
    }

    template<typename T, typename S>
    const S &Relation<T, S>::valueAtIndex(NumN i) const {
        return Y(i);
    }

    template<typename T, typename S>
    NumN Relation<T, S>::has(const T &x) const {
        for (NumN i = 1; i <= X.size(); ++i) {
            if (X(i) == x) {
                return i;
            }
        }
        return 0;
    }

    // It will still add if (x, y) exists
    template<typename T, typename S>
    void Relation<T, S>::add(const T &x, const S &y) {
        X.push(x);
        Y.push(y);
    }

    template<typename T, typename S>
    void Relation<T, S>::add(const Seqce<T> &X, const S &y) {
        for (NumN i = 1; i <= X.size(); ++i) {
            add(X(i), y);
        }
    }

    template<typename T, typename S>
    void Relation<T, S>::clear() {
        X.clear();
        Y.clear();
    }

    template<typename T, typename S>
    void Relation<T, S>::log(const char *s) const {
        log(std::cout, s);
    }

    template<typename T, typename S>
    void Relation<T, S>::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = "";
        }
        io << "Relation " << s << ":\n";
        for (NumN i = 1; i <= size(); ++i) {
            io << "(" << X(i) << ", " << Y(i) << ")\n";
        }
    }
}