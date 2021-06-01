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
#include "files.h"

namespace math21 {
    namespace detail_li {
        void li_test_insertion_sort() {
            Seqce<NumZ> s;
            s.setSize(8);
            s = 55, 66, 33, 44, 77, 88, 11, 22;
            Compare_Num<NumZ> comp;
            insertion_sort(s, s.size(), comp);
            s.log("s", 1);
        }

        void li_test_insertion_sort_2() {
            Seqce<const char *> s;
            s.setSize(4);
            s = "math21!", "I", "love", "Hi,";
            Compare_ss comp;
            insertion_sort(s, s.size(), comp);
            s.log("s2", 1, 0);
        }

        void li_test_insertion_sort_3() {
            Seqce<std::string> s;
            s.setSize(4);
            s = "math21!", "I", "love", "Hi,";
            Compare_string comp;
            insertion_sort(s, s.size(), comp);
            s.log("s3", 1, 0);
        }


    }
}