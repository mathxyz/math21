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

    // should deprecate, and use seqce instead.
    template<typename T>
    struct _Sequence {
    private:
        Seqce <T> v;
    public:
        _Sequence() {
            v.clear();
        }

        virtual ~_Sequence() {
        }

        NumN size() const {
            return v.size();
        }

        NumB isEmpty() const {
            if (size() == 0) {
                return 1;
            }
            return 0;
        }

        void setSize(NumN n) {
            v.setSize(n);
        }

        T &at(NumN i) {
            return v.at(i);
        }

        const T &operator()(NumN j) const {
            return v.operator()(j);
        }

        T &operator()(NumN j) {
            return at(j);
        }

        void add(const T &j) {
            v.push(j);
        }

        void removeLast() {
            v.removeLast();
        }

        void add(const _Sequence<T> &S) {
            for (NumN i = 1; i <= S.size(); ++i) {
                v.push(S(i));
            }
        }

        void copyToNox(_Sequence<T> &S, const T &x) const {
            S.clear();
            for (NumN i = 1; i <= size(); ++i) {
                const T &s = (*this)(i);
                if (s != x)
                    S.add(s);
            }
        }

        void copyTo(_Sequence<T> &S) const {
            S.clear();
            for (NumN i = 1; i <= size(); ++i) {
                S.add((*this)(i));
            }
        }

        NumB contains(const T &x) const {
            for (NumN i = 1; i <= size(); ++i) {
                if (v.operator()(i) == x) {
                    return 1;
                }
            }
            return 0;
        }

        void clear() {
            v.clear();
        }

        void log(const char *s = 0) const {
            log(std::cout, s);
        }

        void log(std::ostream &io, const char *s = 0) const {
            if (s) {
                io << "_Sequence " << s << ":\n";
            }
            v.log(io, 0, 1);
        }

        void toSet(_Set <T> &S) const;
    };

    template<typename T>
    NumB math21_operator_sequence_isEqual(const _Sequence<T> &a, const _Sequence<T> &b) {
        if (a.size() != b.size()) {
            return 0;
        }
        for (NumN i = 1; i <= a.size(); ++i) {
            if (a(i) != b(i)) {
                return 0;
            }
        }
        return 1;
    }

    struct Compare_SequenceR {
        const SequenceR &evs;

        Compare_SequenceR(const SequenceR &evs) : evs(evs) {
        }

        bool operator()(int i1, int i2) {
            NumN i, j;
            i = (NumN) i1 + 1;
            j = (NumN) i2 + 1;

            if (evs(i) < evs(j)) {
                return 1;
            }
            return 0;
        }
    };

    template<typename T>
    std::ostream &operator<<(std::ostream &io, const _Sequence<T> &m) {
        m.log(io);
        return io;
    }

}