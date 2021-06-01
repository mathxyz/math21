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

#include "inner_header.h"

namespace math21 {

    void test_time_random_normal();

    void test_draw_random_normal();

    void test_time_random_uniform();

    void test_draw_random_uniform();

    void test_draw_random_binomial();

    void math21_data_draw_from_sphere_uniformly(MatR &A, NumN n);

    void math21_plot_set_color_red(VecN &color);

    void math21_plot_set_color_green(VecN &color);

    void math21_plot_set_color_blue(VecN &color);

    void math21_plot_set_color(VecN &color);

    void math21_plot_set_color(Seqce <VecN> &colors);

    void math21_plot_point(NumR x, NumR y, TenR &A, const VecN &color);

    void math21_plot_point(NumR x, NumR y, TenR &A, const VecN &color, NumR radius);

    // plot data to image
    void math21_plot_mat_data(const MatR &data, TenR &A, const VecN &color, NumR radius = 0);

    void math21_plot_mat_data_with_option(const MatR &data, TenR &A, const VecN &color, NumR radius= 0, NumN index_i= 0);

    // plot data to image
    void math21_plot_container_data(const Seqce <MatR> &data, TenR &A, const VecN &color, NumR radius = 0);

    void math21_plot_container_with_option(const Seqce <MatR> &data, TenR &A, NumN index_i, NumN index_j);
}