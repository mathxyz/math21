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

#include <cstdarg>
#include "log.h"
#include "log_c.h"
#include "../variables/files.h"

namespace math21 {
    void math21_print(std::ostream &io, const NumN &t, const char *s) {
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    void math21_print(std::ostream &io, const NumZ &t, const char *s) {
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    void math21_print(std::ostream &io, const NumR &t, const char *s) {
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    void math21_print(std::ostream &io, const char &t, const char *s) {
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    void math21_print(std::ostream &io, const char *t, const char *s) {
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    void math21_print(std::ostream &io, const std::string &t, const char *s) {
//        if (s) {
//            io << "name: " << s << std::endl;
//        }
        io << t;
        if (s) {
            io << std::endl;
        }
    }

    // we don't use name s.
    void math21_print(std::ostream &io, const ad::Variable *v, const char *s) {
        v->log(io);
    }

    // we don't use name s.
    void math21_print(std::ostream &io, ad::Variable *v, const char *s) {
        v->log(io, 0, 1);
    }

    void m21log(const char *b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        if (!b) {
            m21cout("null pointer\n");
        } else {
            m21cout(b);
            m21cout("\n");
        }
    }

    const int MATH21_GLOBAL_PARAS_DEBUG_MESSAGE_MAX = 1024;

//    void m21vlog(const char *s, ...) {
//        va_list va;
//        va_start(va, s);
//        char buffer[MATH21_GLOBAL_PARAS_DEBUG_MESSAGE_MAX];
//        vsprintf(buffer, s, va);
//        va_end(va);
//        m21log(buffer);
//    }


    void m21input_getline(std::string s) {
        getline(std::cin, s);
    }

    void m21pause_zzz(const char *s) {
        if (s) {
            m21log(s);
        }
        char b;
        m21input(b);
        std::string str;
        m21input_getline(str);
    }

    void m21pause_pressEnter(const char *s) {
        if (s) {
            m21log(s);
        }
        getchar();
        std::string str;
        m21input_getline(str);
    }

    void math21_tool_exit(const char *s) {
        if (s) {
            m21log(s);
        }
        exit(EXIT_SUCCESS);
    }

    void math21_tool_log_title(const char *name) {
        if (name != 0) {
            std::cout << "######### " << name << " #########" << std::endl;
        } else {
            std::cout << "##################" << std::endl;
        }
    }

    std::string math21_string_get_system_name() {
#if defined(MATH21_WINDOWS)
        return "WINDOWS";
#elif defined(MATH21_WIN_MSVC)
        return "WIN_MSVC";
#elif defined(MATH21_ANDROID)
        return "ANDROID";
#elif defined(MATH21_APPLE)
        return "APPLE";
#elif defined(MATH21_LINUX)
        return "LINUX";
#endif
    }

    void math21_tool_log_num_type() {
        m21log("current system", math21_string_get_system_name());
        m21log("Type used:");
        m21log("NumN", sizeof(NumN));
        m21log("NumZ", sizeof(NumZ));
        m21log("NumR", sizeof(NumR));
        m21log("Type not used:");
        m21log("NumB", sizeof(NumB));
        m21log("NumN8", sizeof(NumN8));
        m21log("NumZ8", sizeof(NumZ8));
        m21log("NumN32", sizeof(NumN32));
        m21log("NumZ32", sizeof(NumZ32));
        m21log("NumN64", sizeof(NumN64));
        m21log("NumZ64", sizeof(NumZ64));
        m21log("int*", sizeof(int *));
        m21log("int", sizeof(int));
        m21log("size_t", sizeof(size_t));
        m21log("long", sizeof(long));
        m21log("float", sizeof(float));
        m21log("double", sizeof(double));
        m21log("long double", sizeof(long double));
        m21log("NumR max", (std::numeric_limits<NumR>::max)());
        m21log("NumR min", (std::numeric_limits<NumR>::min)());
        m21log(
                "\n>>>>> WINDOWS results:\n"
                "Type used:\n"
                "NumN : 4\n"
                "NumZ : 4\n"
                "NumR : 8\n"
                "Type not used:\n"
                "NumB : 4\n"
                "NumN8 : 1\n"
                "NumZ8 : 1\n"
                "NumN32 : 4\n"
                "NumZ32 : 4\n"
                "NumN64 : 8\n"
                "NumZ64 : 8\n"
                "int* : 4\n"
                "int : 4\n"
                "size_t : 4\n"
                "long : 4\n"
                "float : 4\n"
                "double : 8\n"
                "long double : 12\n"
                "NumR max : 1.79769e+308\n"
                "NumR min : 2.22507e-308\n"
                "\n>>>>> LINUX result:\n"
                "Type used:\n"
                "NumN : 4\n"
                "NumZ : 4\n"
                "NumR : 8\n"
                "Type not used:\n"
                "NumB : 4\n"
                "NumN8 : 1\n"
                "NumZ8 : 1\n"
                "NumN32 : 4\n"
                "NumZ32 : 4\n"
                "NumN64 : 8\n"
                "NumZ64 : 8\n"
                "int* : 8\n"
                "int : 4\n"
                "size_t : 8\n"
                "long : 8\n"
                "float : 4\n"
                "double : 8\n"
                "long double : 16\n"
                "NumR max : 1.79769e+308\n"
                "NumR min : 2.22507e-308\n");
    }
}

using namespace math21;

void math21_log_char(const char *name, char x) {
    char s[2];
    s[0] = x;
    s[1] = 0;
    m21log(name, s);
}

void math21_log_int(const char *name, int x) {
    m21log(name, x);
}

void math21_log_size_t(const char *name, size_t x) {
    m21log(name, x);
}

void math21_log_NumN(const char *name, NumN x) {
    m21log(name, x);
}

void math21_log_NumZ(const char *name, NumZ x) {
    m21log(name, x);
}

void math21_log_NumR(const char *name, NumR x) {
    m21log(name, x);
}

void math21_log_flush_stdout() {
    fflush(stdout);
}

void math21_io_save_int(const char *name, int x) {
    FILE *f = fopen(name, "a");
    if (!f) {
        math21_file_error(name);
        return;
    }
    fprintf(f, "%d\n", x);
    fclose(f);
}

void math21_io_save_c_string(const char *name, const char *s) {
    FILE *f = fopen(name, "a");
    if (!f) {
        math21_file_error(name);
        return;
    }
    fprintf(f, "%s\n", s);
    fclose(f);
}
