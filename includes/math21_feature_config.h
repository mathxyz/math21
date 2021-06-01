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

#ifdef MATH21_FLAG_NOT_EXTERNAL
#include "../math21_user_config_generated"
#else
#include "../math21_external_user_config_generated"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// If you are compiling math21 as a shared library and installing it somewhere on your system
// then it is important that any programs that use math21 agree on the state of the
// MATH21_ASSERT statements (i.e. they are either always on or always off).  Therefore,
// uncomment one of the following lines to force all MATH21_ASSERTs to either always on or
// always off.  If you don't define one of these two macros then MATH21_ASSERT will toggle
// automatically depending on the state of certain other macros, which is not what you want
// when creating a shared library.
#ifndef MATH21_FLAG_RELEASE
#define MATH21_DEBUG // debug enabled
#else
//#define MATH21_DISABLE_LOG
#endif


#ifdef MATH21_FLAG_IS_WIN_MSVC
//#ifdef MATH21_ENABLE_ASSERTS
//#undef MATH21_ENABLE_ASSERTS
//#endif
#define MATH21_ENABLE_ASSERTS       // asserts always enabled
#else
#define MATH21_ENABLE_ASSERTS       // asserts always enabled
#endif

//#define MATH21_ENABLE_ASSERTS_REMINDER       // asserts remainder enabled, remind you sth and will make program very slow.
//#define MATH21_DISABLE_ASSERTS // asserts always disabled
#define MATH21_ENABLE_PRINT_TIME


///////////////////////////////// system ////////////////////////////////////
#ifdef MATH21_FLAG_IS_WIN32
#if !defined(MATH21_WINDOWS)
#define MATH21_WINDOWS
#endif
#endif

#ifdef MATH21_FLAG_IS_WIN_MSVC
#if !defined(MATH21_WIN_MSVC)
#define MATH21_WIN_MSVC
#endif
#endif

#ifdef MATH21_FLAG_IS_ANDROID
#if !defined(MATH21_ANDROID)
#define MATH21_ANDROID
#endif
#endif

#ifdef MATH21_FLAG_IS_APPLE
#if !defined(MATH21_APPLE)
#define MATH21_APPLE
#endif
#endif

#ifdef MATH21_FLAG_IS_LINUX
#if !defined(MATH21_LINUX)
#define MATH21_LINUX
#endif
#endif

///////////////////////////////// algorithm ////////////////////////////////////

//#define MATH21_FLAG_UNDERSTANDABLE
#define MATH21_ENABLE_ML

///////////////////////////////// check ////////////////////////////////////
#if defined(MATH21_FLAG_USE_CUDA)
#elif defined(MATH21_FLAG_USE_OPENCL)
#else
#if !defined(MATH21_FLAG_USE_CPU)
#define MATH21_FLAG_USE_CPU
#endif
#endif

#ifdef __cplusplus
}
#endif
