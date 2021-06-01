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

#include "inner_cc.h"

namespace math21 {
    float getticks();

    NumR math21_time_getticks_no_parallel();

    // no gpu
    NumR math21_time_getticks();

#if defined(MATH21_ENABLE_PRINT_TIME)
#define MATH21_PRINT_TIME 1
#else
#define MATH21_PRINT_TIME 0
#endif

/* Macros to simplify time cost print*/
#if MATH21_PRINT_TIME
    static float getticks_time_old, getticks_time_old_2, getticks_average_time_old, getticks_average_time;
    static int getticks_average_time_index;
#define MATH21_PRINT_TIME_START() {getticks_time_old= getticks();}
#define MATH21_PRINT_TIME_END(r) {printf("\ntime cost!!!\n%s: %fs\n", r, getticks() - getticks_time_old);}
#define MATH21_PRINT_TIME_START_2() {getticks_time_old_2= getticks();}
#define MATH21_PRINT_TIME_END_2(r) {printf("\ntime cost!!!\n%s: %fs\n", r, getticks() - getticks_time_old_2);}

#define MATH21_PRINT_AVERAGE_TIME_CREATE() {getticks_average_time= 0.0f;getticks_average_time_index =0;}
#define MATH21_PRINT_AVERAGE_TIME_START() {getticks_average_time_old= getticks();}
#define MATH21_PRINT_AVERAGE_TIME_END(r) {getticks_average_time +=getticks() - getticks_average_time_old;\
getticks_average_time_index++; printf("\naverage time index %d: ,average time %fs\n", getticks_average_time_index, getticks_average_time);}
#define MATH21_PRINT_TIME_DESTROY() {printf("\naverage time %fs over %d times\n",  getticks_average_time/getticks_average_time_index, getticks_average_time_index);}
#else
#define MATH21_PRINT_TIME_START()
#define MATH21_PRINT_TIME_END(r)
#define MATH21_PRINT_TIME_START_2()
#define MATH21_PRINT_TIME_END_2(r)
#define MATH21_PRINT_AVERAGE_TIME_CREATE()
#define MATH21_PRINT_AVERAGE_TIME_START()
#define MATH21_PRINT_AVERAGE_TIME_END(r)
#define MATH21_PRINT_TIME_DESTROY()
#endif
}