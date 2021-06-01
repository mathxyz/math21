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

m21image math21_image_create_empty(NumN nr, NumN nc, NumN nch);

m21image math21_image_create_image(NumN nr, NumN nc, NumN nch);

m21image math21_image_create_image_int_input(int nr, int nc, int nch);

m21image math21_image_clone_image(m21image image0);

void math21_image_destroy_image(m21image *image);

void math21_image_destroy_image_no_pointer_pass(m21image image);

void math21_image_set_image(m21image m, float s);

void math21_image_pixel_value_set_ignore_error(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val);

void math21_image_pixel_value_set(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val);

NumR32 math21_image_pixel_value_get(const m21image *m, NumN ir, NumN ic, NumN ich);

void math21_image_pixel_add_to(m21image *m, NumN ir, NumN ic, NumN ich, NumR32 val);

void math21_image_embed_image(m21image src, m21image dst, NumN dst_offset_r, NumN dst_offset_c);

m21image math21_image_resize_image(m21image im, NumN nr_dst, NumN nc_dst);

m21image math21_image_resize_with_padding(m21image im, NumN nr_dst, NumN nc_dst);

void math21_image_resize_and_embed(m21image src, int nr_resized, int nc_resized, int offset_r, int offset_c,
                                   m21image canvas);

void math21_image_flip_horizontally(m21image src);

// [start, end]
void math21_image_draw_line_horizontal(m21image image, int ir, int start, int end,
                                       float r, float g, float b);

// [start, end]
void math21_image_draw_line_vertical(m21image image, int ic, int start, int end,
                                     float r, float g, float b);

void
math21_image_draw_box_one(m21image image, int r_start, int r_end, int c_start, int c_end, float r, float g, float b);

void math21_image_draw_box_with_width(m21image image, int r_start, int r_end, int c_start, int c_end,
                                      int line_width, float r, float g, float b);

void math21_image_convert_ch_r_c_to_r_c_ch_NumR_2_NumN8(const NumR *src, NumN8 *dst, NumN nr, NumN nc, NumN nch);

void math21_image_convert_r_c_ch_to_ch_r_c_NumN8_2_NumR(const NumN8 *src, NumR *dst, NumN nr, NumN nc, NumN nch);

NumR math21_image_get_ith_color_normalized(NumN ich, NumN i, NumN n);

NumN math21_image_color_get_red(NumN i, NumN n);

NumN math21_image_color_get_green(NumN i, NumN n);

NumN math21_image_color_get_blue(NumN i, NumN n);

#ifdef __cplusplus
}
#endif
