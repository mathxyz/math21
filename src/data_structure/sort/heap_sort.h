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

    namespace detail_li {

        /*
        * node start breaks heap property.
        * n is heap size
        * we just make child tree heap, where node start is child tree root.
        * So size n tree may not be heap.
        * f value {-1, 0, 1}
        */
        template<template<typename> class Container, typename T, typename Compare>
        void satisfyHeapAsRootNode(Container<T> &s, NumZ start, NumZ n,
                                   const Compare &f) {
            NumZ root = start, lc, big;
            while ((lc = 2 * root) <= n) {
                big = root;
                if (f.compare(s(big), s(lc)) < 0)
                    big = lc;
                if (lc + 1 <= n && f.compare(s(big), s(lc + 1)) < 0)
                    big = lc + 1;
                if (big != root) {
                    m21_swap(s.at(big), s.at(root));
                    root = big;
                } else {
                    break;
                }
            }
        }


        /*
     * heapify from bottom to top
     * par is internal node of tree
     * par decrease one every time.
     * n is numbers
     * we grow heap from tree bottom to tree root.
     */
        template<template<typename> class Container, typename T, typename Compare>
        void heapify(Container<T> &s, NumZ n, const Compare &f) {
            NumZ par = n / 2;
            while (par >= 1) {
                satisfyHeapAsRootNode(s, par, n, f);
                par--;
            }
        }

        /*
        * use max heap to realize sort from small to big.
        * T is Num, pointer, or std::string type.
        * n <= s.size()
        */
        //Template template class
        template<template<typename> class Container, typename T, typename Compare>
        void liheapsort(Container<T> &s, NumN n, const Compare &f) {
            MATH21_ASSERT(n <= s.size())
            if (n <= 1)return;
            heapify(s, n, f);
            m21_swap(s(1), s(n));
            n = n - 1;
            while (n > 1) {
                satisfyHeapAsRootNode(s, 1, n, f);
                m21_swap(s(1), s(n));
                n = n - 1;
            }
        }

        template<typename T>
        struct Compare_Num {
            Compare_Num() {
            }

            NumZ compare(const T &a, const T &b) const {
                if (a == b) {
                    return 0;
                } else if (a > b) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

        template<typename T>
        struct Compare_Num_reverse {
            Compare_Num_reverse() {
            }

            NumZ compare(const T &a, const T &b) const {
                if (a == b) {
                    return 0;
                } else if (a < b) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

        struct Compare_ss {
            Compare_ss() {
            }

            NumZ compare(const char *s, const char *t) const {
                const char *s1, *s2;
                s1 = s;
                s2 = t;
                while (*s1 != '\0' && *s1 == *s2) {
                    s1++;
                    s2++;
                }
                return (*s1 - *s2);
            }
        };

        struct Compare_string {
            Compare_string() {
            }

            NumZ compare(const std::string &s, const std::string &t) const {
                const char *s1, *s2;
                s1 = s.c_str();
                s2 = t.c_str();
                while (*s1 != '\0' && *s1 == *s2) {
                    s1++;
                    s2++;
                }
                return (*s1 - *s2);
            }
        };

        struct Compare_string_reverse {
            Compare_string_reverse() {
            }

            NumZ compare(const std::string &s, const std::string &t) const {
                const char *s1, *s2;
                s1 = s.c_str();
                s2 = t.c_str();
                while (*s1 != '\0' && *s1 == *s2) {
                    s1++;
                    s2++;
                }
                return (*s2 - *s1);
            }
        };

        // length + alphabet
        struct Compare_string_by_length {
            NumZ compare(const std::string &s, const std::string &t) const {
                if (s.length() < t.length()) {
                    return -1;
                } else if (s.length() > t.length()) {
                    return 1;
                }
                const char *s1, *s2;
                s1 = s.c_str();
                s2 = t.c_str();
                while (*s1 != '\0' && *s1 == *s2) {
                    s1++;
                    s2++;
                }
                return (*s1 - *s2);
            }
        };

        // sort index, not search index
        template<typename T, template<typename> class Container=Seqce>
        struct Compare_index {
            const Container<T> &s;

            explicit Compare_index(const Container<T> &s) : s(s) {
            }

            NumZ compare(NumN a, NumN b) const {
                if (s(a) == s(b)) {
                    return 0;
                } else if (s(a) > s(b)) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

        template<typename T, template<typename> class Container=Seqce,
                template<typename> class Container2=Seqce>
        struct Compare_index_index {
            const Container<T> &s;
            const Container2<NumN> &x;

            explicit Compare_index_index(const Container<T> &s, const Container2<NumN> &x) : s(s), x(x) {
            }

            NumZ compare(NumN a, NumN b) const {
                if (s(x(a)) == s(x(b))) {
                    return 0;
                } else if (s(x(a)) > s(x(b))) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

        void li_test_heapsort();

        void li_test_heapsort_2();

        void li_test_heapsort_3();
    }


    template<typename T, template<typename> class Container, typename Compare=detail_li::Compare_Num<T> >
    void math21_algorithm_sort(Container<T> &s, const Compare &f = Compare()) {
        detail_li::liheapsort(s, s.size(), f);
    }

    enum {
        sort_type_string_normal = 0,
        sort_type_string_reverse,
        sort_type_string_by_length,
    };

    template<template<typename> class Container>
    void math21_string_sort(Container<std::string> &s, NumN sort_type = 0) {
        if (sort_type == sort_type_string_normal) {
            math21_algorithm_sort(s, detail_li::Compare_string());
        } else if (sort_type == sort_type_string_reverse) {
            math21_algorithm_sort(s, detail_li::Compare_string_reverse());
        } else if (sort_type == sort_type_string_by_length) {
            math21_algorithm_sort(s, detail_li::Compare_string_by_length());
        } else {
            math21_tool_assert(0);
        }
    }


    template<typename T, template<typename> class Container1,
            template<typename> class Container2>
    void math21_algorithm_sort_indexes(const Container1<T> &s, Container2<NumN> &idx) {
        idx.setSize(s.size());
        idx.letters();
        detail_li::Compare_index<T, Container1> comp(s);
        math21_algorithm_sort(idx, comp);
    }

    template<typename T, template<typename> class Container>
    NumN math21_operator_argmin_li(const Container<T> &s) {
        Seqce<NumN> idx;
        math21_algorithm_sort_indexes(s, idx);
        return idx(1);
    }

    template<typename T, template<typename> class Container>
    NumN math21_operator_argmax_li(const Container<T> &s) {
        Seqce<NumN> idx;
        math21_algorithm_sort_indexes(s, idx);
        return idx(idx.size());
    }
}