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
#include "set.h"

namespace math21 {
    struct InversibleMap {
    private:
        // Todo: convert sequence to set.
        // will deprecate.
        Sequence a;
        Sequence b;

        // tmp, but will use eventually.
        Set A;
        Set B;

        void init();

        NumB isInversible() const;

        void destroy();

    public:

        InversibleMap();

        virtual ~InversibleMap();

        NumN size() const;

        NumB isEmpty() const;

        const Sequence &getX() const;

        const Sequence &getY() const;

        Set &getXSet();

        Set &getYSet();


        NumB get_y(NumN x, NumN &y) const;

        NumB get_x(NumN y, NumN &x) const;

        NumN valueAt(NumN x) const;

        NumN value_at_x(NumN x) const;

        NumN value_at_y(NumN y) const;

        NumB has_x(NumN x) const;

        NumB has_y(NumN y) const;

        void add(NumN x, NumN y);

        void remove(NumN x);

        // we don't check if this map is inversible.
        void add(const Set &X, const Set &Y);

        void clear();

        void restrictTo(const Set &X, InversibleMap &dst);

        void copyTo(InversibleMap &g) const;

        void log(const char *s = 0) const;

        void log(std::ostream &io, const char *s = 0) const;

    };

    NumB math21_operator_map_identity(const InversibleMap &f);

    NumB math21_operator_map_permutation(const InversibleMap &f);
}