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

#include <cassert>
#include <sstream>
#include "config/config.h"
#include "error.h"
#include "_assert_c.h"

#define MATH21_NO_EXCEPTIONS

namespace math21 {

#if defined MATH21_DISABLE_ASSERTS
    // if MATH21_DISABLE_ASSERTS is on then never enable MATH21_ASSERT no matter what.
#undef MATH21_ENABLE_ASSERTS
#endif

#if !defined(MATH21_DISABLE_ASSERTS) && (defined MATH21_DEBUG)

    // make sure MATH21_ENABLE_ASSERTS is defined if we are indeed using them.
#ifndef MATH21_ENABLE_ASSERTS
#define MATH21_ENABLE_ASSERTS
#endif

    //#ifdef MATH21_ENABLE_ASSERTS
    //#undef MATH21_ENABLE_ASSERTS
    //#endif

#endif


// -----------------------------

#ifdef __GNUC__
    // There is a bug in version 4.4.5 of GCC on Ubuntu which causes GCC to segfault
    // when __PRETTY_FUNCTION__ is used within certain templated functions.  So just
    // don't use it with this version of GCC.
#  if !(__GNUC__ == 4 && __GNUC_MINOR__ == 4 && __GNUC_PATCHLEVEL__ == 5)
#    define MATH21_FUNCTION_NAME __PRETTY_FUNCTION__
#  else
#    define MATH21_FUNCTION_NAME "unknown function"
#  endif
#elif defined(_MSC_VER)
#define MATH21_FUNCTION_NAME __FUNCSIG__
#else
#define MATH21_FUNCTION_NAME "unknown function"
#endif


    template<typename ExpType, typename MsgType>
    inline void
    _math21_assert_print(const ExpType &expType, const MsgType &msgType, const char *file, const int line,
                         const char *func) {
        if (!(expType)) {
            math21_assert_breakpoint();
            std::ostringstream io;
            io << "\n\nEEError detected at line " << line << ".\n";
            io << "Error detected in file " << file << ".\n";
            io << "Error detected in function " << func << ".\n\n";
            io << "Failing expression was " << expType << ".\n";
            io << std::boolalpha << msgType << "\n";
            printf("%s\n", io.str().c_str());
            assert(0);
//            throw math21::fatal_error(math21::EBROKEN_ASSERT, io.str());
        }
    }

///////////////////////////////// CUDA ////////////////////////////////////
#ifdef MATH21_FLAG_USE_CUDA

    void _math21_assert_cuda(void* err, const char *file, int line,
                                           const char *func);

#ifndef MATH21_ASSERT_CUDA_CALL
#define MATH21_ASSERT_CUDA_CALL(expr)  _math21_assert_cuda(expr, __FILE__, __LINE__, MATH21_FUNCTION_NAME)
#endif
#endif

/////////////////////////////////   ////////////////////////////////////

// todo: add assert(1) to force user to add ";" at the end.
#define MATH21M_CASSERT(_exp, _message)                                              \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        math21_assert_breakpoint();                                           \
        std::ostringstream io;                                       \
        io << "\n\nEEError detected at line " << __LINE__ << ".\n";    \
        io << "Error detected in file " << __FILE__ << ".\n";      \
        io << "Error detected in function " << MATH21_FUNCTION_NAME << ".\n\n";      \
        io << "Failing expression was " << #_exp << ".\n";           \
        io << std::boolalpha << _message << "\n";                    \
        printf("%s\n", io.str().c_str());                    \
        assert(0);      \
    }}

//    throw math21::fatal_error(math21::EBROKEN_ASSERT,io.str());
//    }}

//#define MATH21M_CASSERT(_exp, _message) _math21_assert_print(_exp, _message, __FILE__, __LINE__, MATH21_FUNCTION_NAME);

////
// Make it so the 2nd argument of MATH21_CASSERT is optional.  That is, you can call it like
// MATH21_CASSERT(exp) or MATH21_CASSERT(exp,message).
//https://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion
#define MATH21_MACRO_GLUE(x, y) x y

#define MATH21_MACRO_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, count, ...) count
#define MATH21_MACRO_EXPAND_ARGS(args) MATH21_MACRO_RETURN_ARG_COUNT args
#define MATH21_MACRO_COUNT_ARGS_MAX5(...) MATH21_MACRO_EXPAND_ARGS((__VA_ARGS__, 5, 4, 3, 2, 1, 0))

#define MATH21_MACRO_OVERLOAD_MACRO2(name, count) name##count
#define MATH21_MACRO_OVERLOAD_MACRO1(name, count) MATH21_MACRO_OVERLOAD_MACRO2(name, count)
#define MATH21_MACRO_OVERLOAD_MACRO(name, count) MATH21_MACRO_OVERLOAD_MACRO1(name, count)

#define MATH21_MACRO_CALL_OVERLOAD(name, ...) MATH21_MACRO_GLUE(MATH21_MACRO_OVERLOAD_MACRO(name, MATH21_MACRO_COUNT_ARGS_MAX5(__VA_ARGS__)), (__VA_ARGS__))
////

#define _MATH21_CASSERT1(exp) MATH21M_CASSERT(exp,"")
#define _MATH21_CASSERT2(exp, message) MATH21M_CASSERT(exp,message)
#define MATH21_CASSERT(...) MATH21_MACRO_CALL_OVERLOAD(_MATH21_CASSERT, __VA_ARGS__)

#ifdef MATH21_ENABLE_ASSERTS

#if defined(MATH21_DEBUG)
#define MATH21_ASSERT_INDEX(...) MATH21_CASSERT(__VA_ARGS__)
#else
#define MATH21_ASSERT_INDEX(...) {}
#endif

#define MATH21_ASSERT(...) MATH21_CASSERT(__VA_ARGS__)
//#define MATH21_ASSERT_NOT_CALL(...) MATH21_CASSERT(__VA_ARGS__)
// Todo: comment out.
#define MATH21_ASSERT_NOT_CALL(...) {}
    // this assert will slow running. So we can comment out to speed.
#define MATH21_ASSERT_SLOW(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_PASS(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_ASSERT_FINITE(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_ASSERT_POSITIVE(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_ASSERT_CHECK_VALUE_TMP(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_ASSERT_CHECK_VALUE(...) MATH21_CASSERT(__VA_ARGS__)
    // assert we write the good code. Once passed, we can comment this out.
#define MATH21_ASSERT_CODE(...) MATH21_CASSERT(__VA_ARGS__)
#define MATH21_IF_ASSERT(exp) exp

#else

#define MATH21_ASSERT_INDEX(...) {}
#define MATH21_ASSERT(...) {}
#define MATH21_ASSERT_NOT_CALL(...) {}
#define MATH21_ASSERT_SLOW(...) {}
#define MATH21_PASS(...) {}
#define MATH21_ASSERT_FINITE(...) {}
#define MATH21_ASSERT_POSITIVE(...) {}
#define MATH21_ASSERT_CHECK_VALUE_TMP(...) {}
#define MATH21_ASSERT_CHECK_VALUE(...) {}
#define MATH21_ASSERT_CODE(...) {}
#define MATH21_IF_ASSERT(exp) {}
#endif

#ifdef MATH21_ENABLE_ASSERTS_REMINDER
#define MATH21_ASSERT_REMINDER(...) MATH21_ASSERT(__VA_ARGS__)
#else
#define MATH21_ASSERT_REMINDER(...) {}
#endif

}