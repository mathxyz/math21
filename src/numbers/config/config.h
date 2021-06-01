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

#include "../../../includes/math21_feature_config.h"

///////////////////////////////// OPENMP ////////////////////////////////////
#ifdef MATH21_FLAG_USE_OPENMP

#include <omp.h>
//#pragma omp critical

#if !defined(MATH21_FLAG_IS_PARALLEL)
#define MATH21_FLAG_IS_PARALLEL
#endif

#endif

///////////////////////////////// ANDROID ////////////////////////////////////
#ifdef MATH21_ANDROID

#include <android/log.h>

#ifdef printf
#undef printf
#endif
#define printf(...) __android_log_print(ANDROID_LOG_INFO,"math21",__VA_ARGS__)
#ifdef fprintf
#undef fprintf
#endif
#define fprintf(X, ...) __android_log_print(ANDROID_LOG_INFO,"math21",__VA_ARGS__)
#endif