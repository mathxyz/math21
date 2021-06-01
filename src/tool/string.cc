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
#include "../matrix_op/files.h"
#include "string_c.h"
#include "./string.h"

namespace math21 {
    void math21_string_argv_2_string(int &argc, const char **&argv, Seqce<std::string> &strs, NumN offset) {
        NumZ n = argc - offset;
        if (n <= 0) {
            strs.clear();
            return;
        }
        strs.setSize((NumN) n);
        for (NumN i = 1; i <= strs.size(); ++i) {
            strs(i) = argv[i - 1 + offset];
        }
    }

    void math21_string_2_argv(const Seqce<std::string> &strs, int &argc, const char **&argv, NumN offset) {
        argc = (int) strs.size() - (int) offset;
        if (argc <= 0) {
            argc = 0;
            return;
        }
        argv = (const char **) (new char *[argc]);
        for (NumN i = 1 + offset; i <= strs.size(); ++i) {
            const std::string &str = strs(i);
            auto *cstr = new char[str.length() + 1];
            strcpy(cstr, str.c_str());
            argv[i - 1] = cstr;
        }
    }

    void math21_string_2_argv(const Seqce<std::string> &strs, int &argc, char **&argv, NumN offset) {
        math21_string_2_argv(strs, argc, (const char **&) argv, offset);
    }

    void math21_string_log_argv(const int &argc, const char **argv) {
        m21log("argc", argc);
        m21log("argv:");
        for (NumN i = 0; i < argc; ++i) {
            m21log(argv[i]);
        }
    }

    void math21_string_log_argv(const int &argc, char **argv) {
        math21_string_log_argv(argc, (const char **) argv);
    }

    void math21_string_free_argv(const int &argc, const char **&argv) {
        for (NumN i = 0; i < argc; ++i) {
            delete[] argv[i];
        }
        delete[] argv;
    }

    void math21_string_free_argv(const int &argc, char **&argv) {
        math21_string_free_argv(argc, (const char **&) argv);
    }

    NumB math21_string_find_first(const char *s, char c, NumN &index) {
        if (!s) {
            return 0;
        }
        NumB found = 0;
        NumN i = 0;
        while (s[i] != '\0') {
            if (s[i] == c) {
                index = i;
                found = 1;
                break;
            }
            ++i;
        }
        return found;
    }

    NumB math21_string_find_last(const char *s, char c, NumZ &index0) {
        NumZ i, index;
        i = 0;
        index = -1;
        index0 = 0;
        MATH21_ASSERT(s != 0 && "s is null");
        while (s[i] != '\0') {
            if (s[i] == c) {
                index = i;
            }
            ++i;
        }
        if (index != -1) {
            index0 = index;
            return 1;
        }
        return 0;
    }

    // if not found, it is assumed to be at last.
    void math21_string_split_by_first_separator(const std::string &path, std::string &s1, std::string &s2) {
        NumN pos;
        s1 = path;
        s2 = "";
        NumB found = math21_string_find_first(path.c_str(), '/', pos);
        if (found) {
            s1 = path.substr(0, pos);
            if (pos < path.length()) {
                s2 = path.substr(pos + 1);
            }
        }
    }

    void math21_operator_tensor_from_string(TenN8 &A, NumN n, const NumN8 *s, NumB shallow) {
        math21_operator_tensor_set_data(A, n, s, shallow);
    }

    void math21_operator_tensor_from_string(TenN8 &A, const std::string &s, NumB shallow) {
        math21_operator_tensor_set_data(A, s.size(), (const NumN8 *) s.c_str(), shallow);
    }

    void math21_operator_tensor_to_string(const TenN8 &A, std::string &s) {
        math21_tool_std_string_resize(s, A.size());
        math21_memory_tensor_data_copy_to_buffer(A, reinterpret_cast<NumN8 *>(&s[0]));
    }

    void math21_string_replace(VecN8 &x, NumN8 c, NumN8 t) {
        VecN8 mask;
        math21_op_ele_is_equal(c, x, mask);
        math21_op_set_using_mask(t, x, mask);
    }

    void math21_string_replace(const VecN8 &x, VecN8 &y, const VecN8 &s, const VecN8 &t) {
        math21_operator_vector_replace(x, y, s, t);
    }
}

using namespace math21;

NumN math21_string_length(const char *s) {
    NumN i = 0;
    MATH21_ASSERT(s)
    while (s[i] != '\0')
        ++i;
    return i;
}

NumB _math21_string_get_file_name(const char *s, char *buffer, NumN bufferSize, NumB isRemoveSuffix) {
    NumZ index;
    NumZ s_size = math21_string_length(s);
    NumB flag = math21_string_find_last(s, '/', index);

    // if not find, we return s directly.
    if (flag == 0) {
        return 0;
    }

    NumZ nameSize = s_size - 1 - index;

    // if find, but at end, return empty.
    if (nameSize == 0) {
        return 0;
    }

    if (nameSize + 1 > bufferSize) {
        printf("buffer not enough!\n");
        return 0;
    }
    memcpy(buffer, s + index + 1, nameSize);
    buffer[nameSize] = 0;

    if (isRemoveSuffix) {
        flag = math21_string_find_last(buffer, '.', index);

        // if not find, we return s directly.
        if (flag == 0) {
            return 1;
        }
        buffer[index] = 0;
    }
    return 1;
}

NumB math21_string_get_file_name(const char *s, char *buffer, NumN bufferSize) {
    return _math21_string_get_file_name(s, buffer, bufferSize, 0);
}

NumB math21_string_get_file_name_without_suffix(const char *s, char *buffer, NumN bufferSize) {
    return _math21_string_get_file_name(s, buffer, bufferSize, 1);
}

void math21_string_strip(char *s) {
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n') ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

// note: file must be ascii text.
unsigned char *math21_string_read_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    MATH21_ASSERT(fp, "Failed to open file " << filename);
    size_t size;

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *text = (unsigned char *) math21_vector_calloc_cpu(size + 1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);
    return text;
}

void math21_string_c_free(unsigned char *text) {
    free(text);
}

const char *math21_string_create_from_string(const char *s) {
    if (!s) {
        return 0;
    }
    NumSize n = math21_string_length(s); // n>=0
    return (const char *) math21_vector_create_from_cpuvector_byte_cpu(n + 1, (const NumN8 *) s, 1);
}
