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
#include "../distributions.h"

namespace math21 {

    namespace detail_li {

        // T is Num, pointer, or std::string type.
        // return index of s where found x, or 0 not found.
        // T and S may not be same.
        template<template<typename> class Container, typename T>
        void shuffle(Container<T> &s, NumN n, NumN seed = 21) {
            MATH21_ASSERT(n <= s.size())
            if (n == 0) {
                return;
            }
            DefaultRandomEngine engine(seed);
            RanUniform ranUniform(engine);
//            RanUniform ranUniform;

            NumN t;
            for (NumN i = n; i >= 1; --i) {
                ranUniform.set(1, i);
                math21_random_draw(t, ranUniform);
                if (t != i) {
                    m21_swap(s.at(t), s.at(i));
                }
            }
        }

        template<template<typename> class Container, typename T>
        void shuffle_with_engine(Container<T> &s, NumN n, think::RandomEngine &engine) {
            MATH21_ASSERT(n <= s.size())
            if (n == 0) {
                return;
            }
            RanUniform ranUniform(engine);
            NumN t;
            for (NumN i = n; i >= 1; --i) {
                ranUniform.set(1, i);
                math21_random_draw(t, ranUniform);
                if (t != i) {
                    m21_swap(s.at(t), s.at(i));
                }
            }
        }

        template<template<typename> class Container, typename T>
        void deshuffle(Container<T> &s, NumN n, NumN seed = 21) {
            MATH21_ASSERT(n <= s.size())
            if (n == 0) {
                return;
            }
            DefaultRandomEngine engine(seed);
            RanUniform ranUniform(engine);

            VecN idx(n);
            NumN t;
            for (NumN i = n; i >= 1; --i) {
                ranUniform.set(1, i);
                math21_random_draw(t, ranUniform);
                idx(i) = t;
            }
            for (NumN i = 1; i <= n; ++i) {
                m21_swap(s.at(idx(i)), s.at(i));
            }
        }


        template<template<typename> class Container, typename T>
        void deshuffle_with_engine(Container<T> &s, NumN n, think::RandomEngine &engine) {
            MATH21_ASSERT(n <= s.size())
            if (n == 0) {
                return;
            }
            RanUniform ranUniform(engine);

            VecN idx(n);
            NumN t;
            for (NumN i = n; i >= 1; --i) {
                ranUniform.set(1, i);
                math21_random_draw(t, ranUniform);
                idx(i) = t;
            }
            for (NumN i = 1; i <= n; ++i) {
                m21_swap(s.at(idx(i)), s.at(i));
            }
        }


        void li_test_shuffle();

        void li_test_shuffle_2();
    }

    template<template<typename> class Container, typename T>
    void math21_algorithm_shuffle(Container<T> &s, NumN n, think::RandomEngine &engine) {
        detail_li::shuffle_with_engine(s, n, engine);
    }


    template<typename T, template<typename> class Container>
    NumN math21_operator_container_argmin_random(const Container<T> &s, think::RandomEngine &engine) {
        SeqceN x(s.size());
        SeqceN y(s.size());
        x.letters();
        y.letters();

        math21_algorithm_shuffle(x, x.size(), engine);
        detail_li::Compare_index_index<T, Container> comp(s, x);
        math21_algorithm_sort(y, comp);
        return x(y(1));
    }

    template<typename T, template<typename> class Container>
    NumN math21_operator_container_argmax_random(const Container<T> &s, think::RandomEngine &engine) {
        SeqceN x(s.size());
        SeqceN y(s.size());
        x.letters();
        y.letters();

        math21_algorithm_shuffle(x, x.size(), engine);
        detail_li::Compare_index_index<T, Container> comp(s, x);
        math21_algorithm_sort(y, comp);
        return x(y(y.size()));
    }

    template<typename T, template<typename> class Container>
    NumN math21_operator_container_arg_random(const Container<T> &s, think::RandomEngine &engine) {
        RanUniform ranUniform(engine);
        MATH21_ASSERT(!s.isEmpty())
        ranUniform.set(1, s.size());
        NumN t;
        math21_random_draw(t, ranUniform);
        return t;
    }


}