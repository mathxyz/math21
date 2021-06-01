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

#include "inner.h"

namespace math21 {

    // translate point
    void math21_la_2d_matrix_translate(NumR x, NumR y, MatR &T);

    void math21_la_2d_matrix_translate_reverse_mode(NumR x, NumR y, MatR &T);

    // scale point by (x, y)
    void math21_la_2d_matrix_scale(NumR x, NumR y, MatR &T);

    void math21_la_2d_matrix_scale_reverse_mode(NumR x, NumR y, MatR &T);

    void math21_la_2d_matrix_scale_and_translate(NumR s1, NumR s2, NumR t1, NumR t2, MatR &T);

    void math21_la_2d_matrix_translate_and_scale(NumR s1, NumR s2, NumR t1, NumR t2, MatR &T);

    // rotate point
    void math21_la_2d_matrix_rotate(NumR theta, MatR &T);

    void math21_la_2d_matrix_rotate_reverse_mode(NumR theta, MatR &T);

    // shear x along x-axis, y along y-axis
    void math21_la_2d_matrix_shearing(NumR x, NumR y, MatR &T);

    // shear x along x-axis, y along y-axis
    void math21_la_2d_matrix_shearing_reverse_mode(NumR x, NumR y, MatR &T);

    void math21_la_2d_image_resize(const TenR &A, TenR &B);

    void math21_la_2d_matrix_test(const TenR &A, TenR &B);

    void math21_la_3d_matrix_translate(NumR x, NumR y, NumR z, MatR &T);

    void math21_la_3d_matrix_translate_reverse_mode(NumR x, NumR y, NumR z, MatR &T);

    void math21_la_3d_matrix_translate(const VecR &t, MatR &T);

    void math21_la_3d_matrix_K(NumR fx, NumR fy, NumR u0, NumR v0, MatR &T);

    void math21_la_3d_matrix_KRt(const MatR &K, const MatR &R, const MatR &_t, MatR &P);

    void math21_la_3x4_matrix_from_3d(const MatR &A, MatR &P);

    void math21_la_3x4_matrix_KRt(const MatR &K, const MatR &R, const MatR &t, MatR &P);

    void math21_la_3d_matrix_scale(NumR x, NumR y, NumR z, MatR &T);

    void math21_la_3d_matrix_scale_reverse_mode(NumR x, NumR y, NumR z, MatR &T);

    void math21_la_3d_matrix_rotate_about_x_axis(NumR theta, MatR &T);

    void math21_la_3d_matrix_rotate_about_y_axis(NumR theta, MatR &T);

    void math21_la_3d_matrix_rotate_about_z_axis(NumR theta, MatR &T);

    // deprecate! use the one having math21 prefix instead.
    //! returns 2x3 affine transformation matrix for the planar rotation. angle is in radian
    void getRotationMatrix2D(const VecR &center, NumR angle, MatR &m);

    //! returns 3x3 affine transformation matrix for the planar rotation. angle is in radian
    //! coordinate axes are fixed.
    void math21_la_getRotationMatrix2D(const VecR &center, NumR angle, MatR &m);

    // rotate point p by angle and then translate by b.
    void math21_la_rotate_and_translate_point(const VecR &b, NumR angle, MatR &T);

    //! translate axis to b and then rotate by angle.
    //! T_inv is axis representation of new axis in old coordinate system.
    //! y = Ax+b, x is new coordinate of point p, y is old representation of p.
    //! T_inv = (A, b; 0, 1)
    void math21_la_translate_and_rotate_axis_reverse_mode(const VecR &b, NumR angle, MatR &T_inv);

    void math21_la_rotate_axis(NumR angle, MatR &T);

    void math21_la_3d_rotate_axis_about_x(NumR angle, MatR &T);

    void math21_la_3d_rotate_axis_about_y(NumR angle, MatR &T);

    void math21_la_3d_rotate_axis_about_z(NumR angle, MatR &T);

    void math21_la_3d_post_rotate_axis_about_x(NumR theta, MatR &T);

    void math21_la_3d_post_rotate_axis_about_y(NumR theta, MatR &T);

    void math21_la_3d_post_rotate_axis_about_z(NumR theta, MatR &T);

    //! (i, j) of B = T * (i, j) of A,
    //! T is affine transformation. y~ = T * x~
    //! T = (A, b; 0, 1), A is linear transformation, y = Ax+b,
    NumB math21_la_affine_transform_image(const MatR &A, MatR &B, const MatR &T);

    void math21_la_2d_affine_transform_image(const MatR &A, MatR &B, const MatR &T);

    void math21_la_2d_affine_transform_image_reverse_mode(const MatR &A, MatR &B, const MatR &T);

    NumB math21_la_3d_perspective_projection_image(const MatR &A, MatR &B, const MatR &T);

    //! (i, j) of A = T * (i, j) of B
    void math21_la_affine_transform_image_reverse_mode(const MatR &A, MatR &B, const MatR &T);

    void math21_la_3d_perspective_projection_image_normal_mode(const MatR &A, MatR &B, const MatR &T);

    // error
    void math21_la_3d_perspective_projection_image_reverse_mode(const MatR &A, MatR &B, const MatR &T);

    NumB math21_geometry_points_homogeneous_to_non(const MatR &X, MatR &Y, NumB ignoreBad = 1);

    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*3.
    NumN math21_geometry_project(const MatR &T, const VecR &x, VecR &X);

    NumB math21_device_geometry_project_2d(const NumR *T, const NumR *x, NumR *X);

    void math21_device_geometry_K_project_2d(NumR *u, NumR *v, NumR xc, NumR yc,
                                             NumR fx, NumR fy, NumR u0, NumR v0);

    void math21_device_geometry_K_inverse_project_2d(NumR u, NumR v, NumR *xc, NumR *yc,
                                                     NumR fx, NumR fy, NumR u0, NumR v0);

    NumB math21_geometry_project_3d(const MatR &T, const VecR &x, VecR &X);

    NumB math21_geometry_project_3x4(const MatR &T, const VecR &x, VecR &X);

    // xs: {*,*; *,*; *,*}
    NumB math21_geometry_project_2d_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs);

    // xs: {*,*,1; *,*,1; *,*,1}
    NumB math21_geometry_project_in_batch(const MatR &T, const MatR &xs, MatR &Xs);

    NumB math21_geometry_project_3d_in_batch(const MatR &T, const MatR &xs, MatR &Xs);

    NumB math21_geometry_project_3d_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs);

    NumB math21_geometry_project_3x4_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs);

    NumB math21_geometry_project_Interval2D(const MatR &T, const Interval2D &I, MatR &vertices);

    // Xab = Xb - Xa. Here T is 3*3.
    NumN math21_geometry_project(const MatR &T, const VecR &xa, const VecR &xb, VecR &Xab);

    NumB geometry_cal_projection(const VecR &x0, const VecR &x1, const VecR &x2, const VecR &x3, MatR &T);

    NumN geometry_cal_projection_scale(const VecR &xa, const VecR &xb, NumR l, MatR &T);

    NumB geometry_cal_projection_scale(const Seqce <VecR> &xas, const Seqce <VecR> &xbs,
                                       const Seqce <NumR> &ls,
                                       MatR &T);

    NumB geometry_cal_projection_to_world(const MatR &Xs, const MatR &xs, MatR &T);

    NumB math21_geometry_project_cal_image_index(const MatR &T, NumN nr, NumN nc, TenR &index);

    void math21_geometry_project_cal_camera_characteristics(const MatR &A, NumN nr, NumN nc,
                                                            NumR &fovx, NumR &fovy,
                                                            NumR &aspectRatio);

    NumB math21_geometry_project_perspective_get_rectangles_from_corrected(NumN nr, NumN nc, const MatR &A,
                                                                           const VecR &distortion,
                                                                           const MatR &R,
                                                                           const MatR &A_new,
                                                                           Interval2D &inner,
                                                                           Interval2D &outer,
                                                                           NumB isUseInner = 1,
                                                                           NumB isUseOuter = 1,
                                                                           NumB isNormalized = 0);

    NumB math21_geometry_project_fisheye_get_rectangles_from_corrected(NumN nr, NumN nc, const MatR &A,
                                                                       const VecR &distortion,
                                                                       const MatR &R,
                                                                       const MatR &A_new,
                                                                       VecR &center_mass,
                                                                       Interval2D &inner,
                                                                       Interval2D &outer,
                                                                       NumB isUseInner = 1,
                                                                       NumB isUseOuter = 1,
                                                                       NumB isNormalized = 0);

    NumB math21_geometry_project_fisheye_points_distorted_to_corrected(const MatR &src, MatR &dst, const MatR &A,
                                                                       const VecR &distortion,
                                                                       const MatR &R,
                                                                       const MatR &A_new,
                                                                       NumB isNormalized);

    NumB math21_geometry_project_perspective_points_distorted_to_corrected(const MatR &src, MatR &dst, const MatR &A,
                                                                           const VecR &distortion,
                                                                           const MatR &R,
                                                                           const MatR &A_new,
                                                                           NumB isNormalized = 0);

    NumB math21_geometry_project_fisheye_get_camera_matrix_combined_with_viewport(const MatR &A,
                                                                                  const VecR &distortion,
                                                                                  NumN nr_old, NumN nc_old,
                                                                                  MatR &A_new,
                                                                                  NumN nr_new, NumN nc_new,
                                                                                  NumR alpha);

    // alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
    // valid) and 1 (when all the source image pixels are retained in the undistorted image).
    // But it can also be outside of [0, 1], just tested. (Todo: prove)
    NumB math21_geometry_project_perspective_get_camera_matrix_combined_with_viewport(const MatR &A,
                                                                                      const VecR &distortion,
                                                                                      NumN nr_old, NumN nc_old,
                                                                                      MatR &A_new,
                                                                                      NumN nr_new, NumN nc_new,
                                                                                      NumR alpha = 0);

    NumB math21_geometry_project_fisheye_cal_corrected_image_index(const MatR &A,
                                                                   const VecR &distortion,
                                                                   const MatR &R,
                                                                   const MatR &A_new,
                                                                   NumN nr, NumN nc, TenR &index);

    NumB math21_geometry_project_perspective_cal_corrected_image_index(const MatR &A,
                                                                       const VecR &distortion,
                                                                       const MatR &R,
                                                                       const MatR &A_new,
                                                                       NumN nr, NumN nc, TenR &index);

    void math21_geometry_convert_project_between_left_and_right_hand(MatR &T);

    void math21_geometry_convert_perspective_distortion_between_left_and_right_hand(VecR &distortion);

    NumB math21_geometry_project_cal_image_index(const MatR &T, NumR x1, NumR x2,
                                                 NumR y1, NumR y2, NumN nr, NumN nc,
                                                 TenR &src_index);

    namespace detail {
        void math21_geometry_warp_image_using_indexes_cpu(const TenR &A, TenR &B, const TenR &index,
                                                      const Interval2D &I, NumB isInterleaved, NumN interpolationFlag);

#ifdef MATH21_FLAG_USE_CUDA

        void math21_geometry_warp_image_using_indexes_cuda(const TenR &A, TenR &B, const TenR &index,
                                                           const Interval2D &I);

#endif
    }

    // I can be empty.
    void math21_geometry_warp_image_using_indexes(const TenR &A, TenR &B, const TenR &index,
                                                  const Interval2D &I, NumB isInterleaved = 0,
                                                  NumN interpolationFlag = m21_flag_interpolation_none);

    void math21_geometry_warp_image_using_project_mat(const MatR &src, MatR &dst,
                                                      const MatR &A, const MatR &distortion, const MatR &R,
                                                      const MatR &P,
                                                      NumN nr_new, NumN nc_new, NumB isInterleaved, NumN interpolationFlag);

    // User should make sure points in A lie in index region.
    void math21_geometry_project_points_using_indexes(const MatR &A, MatR &B, const TenR &index);

    // X = T*x, xs * T_trans = Xs
    NumB math21_geometry_cal_affine(const MatR &xs, const MatR &Xs, MatR &T);

    // (x1,y1), (x2, y2) as diagonal.
    NumB
    math21_geometry_cal_affine_matrix_axis_to_matrix(MatR &T, NumR x1, NumR x2, NumR y1, NumR y2, NumN nr, NumN nc);

    void scale(const MatR &m, MatR &C, NumR a, NumR b, NumR c, NumR d);

    void scale(const TenR &X, TenR &Y, NumR c, NumR d);

    void sample_sine_voice(TenR &X, TenR &Y, NumN seconds);

    void math21_sample_heart(MatR &A);

    void draw_data_at_matrix_axis(const TenR &X_sample, const TenR &Y_sample,
                                  TenN &matrix_axis, NumN axis_x, NumN axis_y);

    // la means linear algebra.
    void la_scale_data_2d(MatR &data,
                          const MatR &M);

    void la_scale_data_2d(const MatR &data, MatR &data_new,
                          const MatR &M);

    void la_scale_data_2d(const MatR &data, MatR &data_new,
                          NumR c1, NumR d1, NumR c2, NumR d2);

    // deprecate, use math21_la_data_2d_bound instead.
    void la_data_2d_bound(const MatR &A,
                          NumR &a1, NumR &b1, NumR &a2, NumR &b2);

    void math21_la_data_2d_bound(const TenR &A,
                                 NumR &a1, NumR &b1, NumR &a2, NumR &b2);

    void math21_la_data_2d_bound(const TenR &A, Interval2D &I);

    void math21_la_data_2d_bound_in_batch(const TenR &A,
                                          NumR &a1, NumR &b1, NumR &a2, NumR &b2);

    void math21_la_data_2d_bound_in_batch(const TenR &A,
                                          Interval2D &I);

    void la_getScaleTranslationMatrix2D(MatR &M,
                                        NumR a1, NumR b1, NumR a2, NumR b2,
                                        NumR c1, NumR d1, NumR c2, NumR d2);

    void math21_la_getScaleTranslationMatrix2D(MatR &M,
                                               const Interval2D &I1,
                                               Interval2D &I2);

    void math21_la_2d_matrix_compute_matrix_axis_to_matrix(NumN axis_x, NumN axis_y, MatR &T);

    void math21_la_2d_matrix_compute_matrix_to_matrix_axis(NumN axis_x, NumN axis_y, MatR &T_final);

    void la_draw_data_2d_at_matrix_axis_no_scale(const MatR &data, TenN &matrix_axis);

    NumB math21_geometry_polygon_include_points(const math21::MatR &polygon, NumR x, NumR y);

    void math21_geometry_linear_triangulation(const MatR &x1s, const MatR &x2s, MatR &Xs,
                                              const MatR &P1, const MatR &P2);
}