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

#include "inner.h"
#include "inversible_map.h"
#include "files.h"

namespace math21 {
    void InversibleMap::init() {
        a.clear();
        b.clear();
    }

    NumB InversibleMap::isInversible() const {
        Set a2;
        a.toSet(a2);
        Set b2;
        b.toSet(b2);
        if (a.size() == a2.size() && b.size() == b2.size()) {
            return 1;
        } else {
            return 0;
        }
    }

    InversibleMap::InversibleMap() {
        init();
    }

    void InversibleMap::destroy() {
        MATH21_ASSERT(isInversible())
        clear();
    }

    InversibleMap::~InversibleMap() {
        destroy();
    }

    NumN InversibleMap::size() const {
        return a.size();
    }

    NumB InversibleMap::isEmpty() const {
        if (size() == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    const Sequence &InversibleMap::getX() const {
        return a;
    }

    Set &InversibleMap::getXSet() {
        a.toSet(A);
        MATH21_ASSERT(a.size() == A.size())
        return A;
    }

    Set &InversibleMap::getYSet() {
        b.toSet(B);
        MATH21_ASSERT(b.size() == B.size())
        return B;
    }

    const Sequence &InversibleMap::getY() const {
        return b;
    }

    NumN InversibleMap::value_at_x(NumN x) const {
        NumN y = 0;
        for (NumN i = 1; i <= a.size(); ++i) {
            if (a(i) == x) {
                y = b(i);
                return y;
            }
        }
        MATH21_ASSERT(0, "x is " << x)
        return y;
    }

    NumN InversibleMap::valueAt(NumN x) const {
        return value_at_x(x);
    }

    NumN InversibleMap::value_at_y(NumN y) const {
        NumN x = 0;
        for (NumN i = 1; i <= a.size(); ++i) {
            if (b(i) == y) {
                x = a(i);
                return x;
            }
        }
        MATH21_ASSERT(0, "y is " << y)
        return x;
    }

    NumB InversibleMap::get_y(NumN x, NumN &y) const {
        for (NumN i = 1; i <= a.size(); ++i) {
            if (a(i) == x) {
                y = b(i);
                return 1;
            }
        }
        return 0;
    }

    NumB InversibleMap::get_x(NumN y, NumN &x) const {
        for (NumN i = 1; i <= a.size(); ++i) {
            if (b(i) == y) {
                x = a(i);
                return 1;
            }
        }
        return 0;
    }

    NumB InversibleMap::has_x(NumN x) const {
        for (NumN i = 1; i <= a.size(); ++i) {
            if (a(i) == x) {
                return 1;
            }
        }
        return 0;
    }

    NumB InversibleMap::has_y(NumN y) const {
        for (NumN i = 1; i <= a.size(); ++i) {
            if (b(i) == y) {
                return 1;
            }
        }
        return 0;
    }

    // Todo: should check
    void InversibleMap::add(NumN x, NumN y) {
        a.add(x);
        b.add(y);
    }

    void InversibleMap::remove(NumN x) {
        if (x == a(size())) {
            a.removeLast();
            b.removeLast();
        } else {
            MATH21_ASSERT(0, "not support in current version")
        }
    }

    // we don't check if this map is inversible.
    // now we check.
    void InversibleMap::add(const Set &X, const Set &Y) {
        MATH21_ASSERT(X.size() == Y.size())
        for (NumN i = 1; i <= X.size(); ++i) {
            add(X(i), Y(i));
        }
        MATH21_ASSERT(getXSet().size() == getYSet().size())
    }

    void InversibleMap::clear() {
        MATH21_ASSERT(getXSet().size() == getYSet().size())
        a.clear();
        b.clear();
    }

    void InversibleMap::restrictTo(const Set &X, InversibleMap &dst) {
        dst.clear();
        for (NumN i = 1; i <= size(); ++i) {
            if (X.contains(a(i))) {
                dst.add(a(i), b(i));
            }
        }
    }

    void InversibleMap::copyTo(InversibleMap &g) const {
        g.clear();
        a.copyTo(g.a);
        b.copyTo(g.b);
    }

    void InversibleMap::log(const char *s) const {
        log(std::cout, s);
    }

    void InversibleMap::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = "";
        }
        io << "InversibleMap " << s << ":\n";
        for (NumN i = 1; i <= size(); ++i) {
            io << "(" << a(i) << ", " << b(i) << ")\n";
        }
    }

    NumB math21_operator_map_identity(const InversibleMap &f) {
        return math21_operator_sequence_isEqual(f.getX(), f.getY());
    }

    NumB math21_operator_map_permutation(const InversibleMap &f) {
        Set A, B;
        f.getX().toSet(A);
        f.getY().toSet(B);
        A.sort();
        B.sort();
        return math21_operator_set_isEqual(A, B);
    }
}