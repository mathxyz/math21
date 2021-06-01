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

#include <cstring>
#include <string>
#include "files.h"
#include "tool2.h"

namespace math21 {
    template<>
    void math21_string_to_type_generic(const std::string &s, std::string &num) {
        num = s;
    }

    NumR math21_string_to_NumR(const std::string &s) {
        std::istringstream iss(s);
        NumR num;
        iss >> num;
        return num;
    }

    NumN math21_string_to_NumN(const std::string &s) {
        std::istringstream iss(s);
        NumN num;
        iss >> num;
        return num;
    }

    NumZ math21_string_to_NumZ(const std::string &s) {
        std::istringstream iss(s);
        NumZ num;
        iss >> num;
        return num;
    }

    std::string math21_string_NumZ_to_string(NumZ x, NumN width) {
        char buf[32];
        std::string format = "%." + math21_string_to_string(width) + "d";
        sprintf(buf, format.c_str(), x);
        return buf;
    }

    std::string math21_string_replicate_n(NumN n, const std::string &s) {
        std::string out = "";
        for (NumN i = 1; i <= n; ++i) {
            out += math21_string_to_string(s);
        }
        return out;
    }

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2) {
        std::string out = s1 + s2;
        return out;
    }

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3) {
        std::string out = s1 + s2 + s3;
        return out;
    }

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                                          const std::string &s4) {
        std::string out = s1 + s2 + s3 + s4;
        return out;
    }

    std::string math21_string_concatenate(const std::string &s1, const std::string &s2, const std::string &s3,
                                          const std::string &s4, const std::string &s5) {
        std::string out = s1 + s2 + s3 + s4 + s5;
        return out;
    }

    NumB math21_string_is_equal(const char *str1, const char *str2) {
        if (strcmp(str1, str2) == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    NumB math21_string_is_equal(const std::string &str1, const std::string &str2) {
        return math21_string_is_equal(str1.c_str(), str2.c_str());
    }

    NumB math21_point_isEqual(const NumR &x, const NumR &y, NumR epsilon) {
        NumR tmp;
        tmp = y - x;
        if (xjabs(tmp) > epsilon) {
            return 0;
        } else {
            return 1;
        }
    }

    NumB math21_point_isEqual(const NumN &x, const NumN &y, NumR epsilon) {
        return math21_point_isEqual((NumR) x, (NumR) y, epsilon);
    }

    NumB math21_point_isEqual(const NumZ &x, const NumZ &y, NumR epsilon) {
        return math21_point_isEqual((NumR) x, (NumR) y, epsilon);
    }

    NumB math21_type_size_t_is_4_bytes() {
        if (sizeof(size_t) == 4) {
            return 1;
        } else {
            return 0;
        }
    }

    NumB math21_type_NumSize_is_4_bytes() {
        if (sizeof(NumSize) == 4) {
            return 1;
        } else {
            return 0;
        }
    }

}