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

#include <iostream>
#include <vector>
#include "inner.h"
#include "SimpleStreamBuf.h"

namespace math21 {
    namespace ad {
        struct Variable;
    }

    void math21_print(std::ostream &io, const NumN &t, const char *s = 0);

    void math21_print(std::ostream &io, const NumZ &t, const char *s = 0);

    void math21_print(std::ostream &io, const NumR &t, const char *s = 0);

    void math21_print(std::ostream &io, const char &t, const char *s = 0);

    void math21_print(std::ostream &io, const char *t, const char *s = 0);

    void math21_print(std::ostream &io, const std::string &t, const char *s = 0);

    template<typename T>
    void math21_print(std::ostream &io, const T &t, const char *s = 0) {
        t.log(io, s);
    }

    void math21_print(std::ostream &io, const ad::Variable *v, const char *s = 0);

    void math21_print(std::ostream &io, ad::Variable *v, const char *s = 0);

    template<typename T1, typename T2>
    void m21_log(std::ostream &io, T1 t1, T2 t2) {
        io << t1 << ", " << t2 << std::endl;
    }

    template<typename T>
    void m21logRepeat(T b, int count = 20) {
        for (int i = 0; i < count; i++) {
            std::cout << b;
        }
        std::cout << std::endl;

    }

    template<typename T>
    void m21cout(const T &a) {
#ifdef MATH21_ANDROID
        SimpleStreamBuf ssb;
        std::streambuf *backup;
        backup = std::cout.rdbuf();
        std::cout.rdbuf(&ssb);
        std::cout << a;
        std::cout.rdbuf(backup);
#else
        std::cout << a;
#endif

    }

    template<typename T>
    void m21fail(T const &message) {
        std::cerr << "Error: " << message << std::endl;
        exit(EXIT_FAILURE);
    }

    template<typename T, typename T2>
    void m21fail(T const &message, T2 t2) {
        std::cerr << "Error: " << message << ", " << t2 << std::endl;
        exit(EXIT_FAILURE);
    }

    template<typename T>
    void m21exit(T const &message) {
        std::cerr << "Exit: " << message << std::endl;
        exit(EXIT_SUCCESS);
    }

    template<typename T>
    void m21error(T const &message) {
        std::cerr << "Error: " << message << std::endl;
    }

    template<typename T>
    void m21warn(T const &message) {
        std::cerr << "Warn: " << message << std::endl;
    }

    template<typename T>
    void m21todo(T const &message) {
        std::cerr << "TODO: " << message << std::endl;
    }

    template<typename T>
    void m21warn(std::string const &message, T b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cerr << "(Warn) " << message << " : " << b << std::endl;
    }

    template<typename T>
    void m21logNoNewLine(T b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif

        std::cout << b;
    }

    void m21log(const char *b);

#define m21vlog printf

//    void m21vlog(const char *s, ...);

    template<typename T>
    void m21log(T b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif

        m21cout(b);
        m21cout("\n");
    }

    template<typename T>
    void m21input(T &b) {
        std::cin >> b;
    }

    void m21input_getline(std::string s);

    void m21pause_zzz(const char *s = 0);

    void m21pause_pressEnter(const char *s = 0);

    void math21_tool_exit(const char *s = 0);

    template<typename T>
    void m21log(const char *message, T b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        if (message) {
            m21cout(message);
        }
        m21cout(" : ");
        m21cout(b);
        m21cout("\n");
    }

    template<typename T>
    void m21log(const std::string &message, T b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif

        m21cout(message);
        m21cout(" : ");
        m21cout(b);
        m21cout("\n");
    }

    template<typename T1, typename T2>
    void m21log(std::string const &message, T1 b, T2 c) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << b << ", " << c << std::endl;
    }


    template<typename T>
    void m21log(std::string const &message, T b, T c, T d) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << b << " : " << c << " : " << d << std::endl;
    }

    template<typename T>
    void m21log(std::string const &message, T a, T b, T c, T d) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << a << " : " << b << " : " << c << " : " << d << std::endl;
    }

    template<typename T>
    void m21log(std::string const &message, T a, T b, T c, T d, T e) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << a << " : " << b << " : " << c << " : " << d << " : " << e << std::endl;
    }

    template<typename T>
    void m21log(std::string const &message, T a, T b, T c, T d, T e, T f) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << a << " : " << b << " : " << c << " : " << d << " : " << e << " : " << f
                  << std::endl;
    }

    template<typename T>
    void m21log(std::string const &message, T a, T b, T c, T d, T e, T f, T g) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << a << " : " << b << " : " << c << " : " << d << " : " << e << " : " << f
                  << " : "
                  << g << std::endl;
    }

    template<typename T>
    void m21log(std::string const &message, T a, T b, T c, T d, T e, T f, T g, T h) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << message << " : " << a << " : " << b << " : " << c << " : " << d << " : " << e << " : " << f
                  << " : "
                  << g << " : " << h << std::endl;
    }

    template<typename T>
    void m21logArray(T b[], int n) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << "{";
        for (int i = 0; i < n; i++) {
            std::cout << b[i];
            if (i < n - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }

    template<typename VecType>
    void m21logContainer(const VecType &b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << "{";
        for (int i = 1; i <= b.size(); ++i) {
            std::cout << b(i);
            if (i < b.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }

    template<typename T>
    void m21log(const std::vector<T> &b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << "{";
        for (int i = 0; i < b.size(); i++) {
            std::cout << b[i];
            if (i < b.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }

    template<typename T>
    void m21log(const std::vector<std::vector<T> > &b) {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        std::cout << "{";
        for (int i = 0; i < b.size(); i++) {
            m21log(b[i]);
            if (i < b.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }

    template<typename T>
    void math21_tool_log_c_array(const T *data, NumN size) {
        for (NumN i = 1; i <= size; ++i) {
            m21log(data[i - 1]);
        }
    }

    void math21_tool_log_title(const char *name = 0);

    void math21_tool_log_num_type();

    std::string math21_string_get_system_name();

#define MATH21_LOG_NAME_VALUE_1_ARGS(a)              std::cout<< #a<<" = "<<a<<";"<<std::endl;
#define MATH21_LOG_NAME_VALUE_2_ARGS(a, b)      std::cout<< #a<<" = "<<b<<";"<<std::endl;
#define MATH21_LOG_NAME_VALUE_GET_3TH_ARG(arg1, arg2, arg3, ...) arg3
#define MATH21_LOG_NAME_VALUE_CHOOSER(...) MATH21_LOG_NAME_VALUE_GET_3TH_ARG(__VA_ARGS__,  MATH21_LOG_NAME_VALUE_2_ARGS, MATH21_LOG_NAME_VALUE_1_ARGS)

#ifdef MATH21_WIN_MSVC
#define MATH21_LOG_NAME_VALUE(...) {}
#else
#define MATH21_LOG_NAME_VALUE(...) MATH21_LOG_NAME_VALUE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#endif

//
#define MATH21_LOG_NAME_VALUE_POINTER_1_ARGS(a)              if(a){std::cout<< "*"<< #a<<" = "<<*a<<";"<<std::endl;}else{std::cout<< #a<<" = null;"<<std::endl;};
#define MATH21_LOG_NAME_VALUE_POINTER_2_ARGS(a, b)      if(b){std::cout<< "*"<< #a<<" = "<<*b<<";"<<std::endl;}else{std::cout<< #a<<" = null;"<<std::endl;};
#define MATH21_LOG_NAME_VALUE_POINTER_GET_3TH_ARG(arg1, arg2, arg3, ...) arg3
#define MATH21_LOG_NAME_VALUE_POINTER_CHOOSER(...) MATH21_LOG_NAME_VALUE_POINTER_GET_3TH_ARG(__VA_ARGS__,  MATH21_LOG_NAME_VALUE_POINTER_2_ARGS, MATH21_LOG_NAME_VALUE_POINTER_1_ARGS)

#ifdef MATH21_WIN_MSVC
#define MATH21_LOG_NAME_VALUE_POINTER(...) {}
#else
#define MATH21_LOG_NAME_VALUE_POINTER(...) MATH21_LOG_NAME_VALUE_POINTER_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#endif

//
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY_1_ARGS(a)              if(a){std::cout<< #a<<" = "<<a<<";"<<std::endl;}else{std::cout<< #a<<" = null;"<<std::endl;};
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY_2_ARGS(a, b)      if(b){std::cout<< #a<<" = "<<b<<";"<<std::endl;}else{std::cout<< #a<<" = null;"<<std::endl;};
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY_GET_3TH_ARG(arg1, arg2, arg3, ...) arg3
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY_CHOOSER(...) MATH21_LOG_NAME_VALUE_POINTER_ONLY_GET_3TH_ARG(__VA_ARGS__,  MATH21_LOG_NAME_VALUE_POINTER_ONLY_2_ARGS, MATH21_LOG_NAME_VALUE_POINTER_ONLY_1_ARGS)

#ifdef MATH21_WIN_MSVC
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY(...) {}
#else
#define MATH21_LOG_NAME_VALUE_POINTER_ONLY(...) MATH21_LOG_NAME_VALUE_POINTER_ONLY_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#endif

#define MATH21_STRINGIFY(x) #x
#define MATH21_CONCATENATOR(x, y) x##y
}