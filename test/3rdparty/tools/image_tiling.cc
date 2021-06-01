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

#include "image_tiling.h"

int test_tiling() {
    tiling_config config;
    config.alpha_m = 0.35f;
    config.alpha_n = 0.35f;
    config.m1 = 972;
    config.n1 = 1296;
    config.m2 = 3;
    config.n2 = 3;
    config.mk_min = 500;
    config.nk_min = 500;
    config.noRedundant = true;
    int size = config.m2 + config.n2 + 2 + 2;
    auto *results = new int[size];
    tiling(&config, results);
    delete[]results;
    return 1;
}

int test_tiling0() {
    tiling_config config;
    config.alpha_m = 0.1f;
    config.alpha_n = 0.1f;
    config.m1 = 2000;
    config.n1 = 1000;
    config.m2 = 1;
    config.n2 = 1;
    config.mk_min = 300;
    config.nk_min = 300;
    config.noRedundant = true;
    int size = config.m2 + config.n2 + 2 + 2;
    auto *results = new int[size];
    tiling(&config, results);
    delete[]results;
    return 1;
}
