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
    void math21_string_argv_2_string(int &argc, const char **&argv, Seqce <std::string> &strs, NumN offset = 0);

    void math21_string_2_argv(const Seqce <std::string> &strs, int &argc, const char **&argv, NumN offset = 0);

    void math21_string_2_argv(const Seqce <std::string> &strs, int &argc, char **&argv, NumN offset = 0);

    void math21_string_log_argv(const int &argc, const char **argv);

    void math21_string_log_argv(const int &argc, char **argv);

    void math21_string_free_argv(const int &argc, const char **&argv);

    void math21_string_free_argv(const int &argc, char **&argv);

    NumB math21_string_find_first(const char *s, char c, NumN &index);

    // s is not null
    // return 0 if not find, 1 success. index will be last index of c.
    NumB math21_string_find_last(const char *s, char c, NumZ &index0);

    void math21_string_split_by_first_separator(const std::string &path, std::string &s1, std::string &s2);

    void math21_operator_tensor_from_string(TenN8 &A, NumN n, const NumN8 *s, NumB shallow = 1);

    void math21_operator_tensor_from_string(TenN8 &A, const std::string &s, NumB shallow = 1);

    void math21_operator_tensor_to_string(const TenN8 &A, std::string &s);

    void math21_string_replace(VecN8 &x, NumN8 c, NumN8 t);

    void math21_string_replace(const VecN8 &x, VecN8 &y, const VecN8 &s, const VecN8 &t);
}