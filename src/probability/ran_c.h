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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

float math21_pr_rand_uniform(float min, float max);

float math21_pr_rand_normal();

void *math21_random_DefaultRandomEngine_create(NumN seed);

void math21_random_DefaultRandomEngine_destroy(void *engine);

void *math21_random_RanUniform_create(void *engine0);

void math21_random_RanUniform_destroy(void *ranUniform);

void math21_random_RanUniform_set(void *ranUniform0, NumR a, NumR b);

void *math21_random_RanNormal_create(void *engine0);

void math21_random_RanNormal_destroy(void *ranNormal);

void math21_random_RanNormal_set(void *ranNormal0, NumR mu, NumR sigma);

NumN32 math21_pr_rand_NumN32();

NumN64 math21_pr_rand_NumN64();

size_t math21_pr_rand_size_t();

NumSize math21_pr_rand_NumSize();

#ifdef __cplusplus
}
#endif
