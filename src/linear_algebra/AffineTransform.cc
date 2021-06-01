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

#include "../op/files.h"
#include "inner_src.h"
#include "AffineTransform.h"

namespace math21 {
    void math21_la_2d_matrix_translate(NumR x, NumR y, MatR &T) {
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        T =
                1, 0, x,
                0, 1, y,
                0, 0, 1;
    }

    void math21_la_2d_matrix_translate_reverse_mode(NumR x, NumR y, MatR &T) {
        math21_la_2d_matrix_translate(-x, -y, T);
    }

    void math21_la_2d_matrix_scale(NumR x, NumR y, MatR &T) {
        MATH21_ASSERT(x != 0 && y != 0)
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        T =
                x, 0, 0,
                0, y, 0,
                0, 0, 1;
    }

    void math21_la_2d_matrix_scale_reverse_mode(NumR x, NumR y, MatR &T) {
        MATH21_ASSERT(x != 0 && y != 0)
        math21_la_2d_matrix_scale(1 / x, 1 / y, T);
    }

    void math21_la_2d_matrix_scale_and_translate(NumR s1, NumR s2, NumR t1, NumR t2, MatR &T) {
        MATH21_ASSERT(s1 != 0 && s2 != 0)
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        T =
                s1, 0, t1,
                0, s2, t2,
                0, 0, 1;
    }

    void math21_la_2d_matrix_translate_and_scale(NumR s1, NumR s2, NumR t1, NumR t2, MatR &T) {
        MATH21_ASSERT(s1 != 0 && s2 != 0)
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        T =
                s1, 0, s1 * t1,
                0, s2, s2 * t2,
                0, 0, 1;
    }

    void math21_la_2d_matrix_rotate(NumR theta, MatR &T) {
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = xjsin(theta);
        c = -xjsin(theta);
        d = xjcos(theta);
        T =
                a, c, 0,
                b, d, 0,
                0, 0, 1;
    }

    void math21_la_2d_matrix_rotate_reverse_mode(NumR theta, MatR &T) {
        math21_la_2d_matrix_rotate(-theta, T);
    }

    // shear x along x-axis, y along y-axis
    void math21_la_2d_matrix_shearing(NumR x, NumR y, MatR &T) {
        MATH21_ASSERT(x * y != 1)
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }

        T =
                1, x, 0,
                y, 1, 0,
                0, 0, 1;
    }

// shear x along x-axis, y along y-axis
    void math21_la_2d_matrix_shearing_reverse_mode(NumR x, NumR y, MatR &T) {
        MATH21_ASSERT(x * y != 1)
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        NumR a, b, c, d, det;
        det = 1 - x * y;
        a = 1 / det;
        b = -y / det;
        c = -x / det;
        d = 1 / det;
        T =
                a, c, 0,
                b, d, 0,
                0, 0, 1;
    }

    void math21_la_2d_image_resize(const TenR &A, TenR &B) {
        MATH21_ASSERT(A.dims() == B.dims())
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        NumN nr;
        NumN nc;
        NumN nr_A, nc_A;
        NumN dims = A.dims();
        if (A.dims() == 3) {
            MATH21_ASSERT(A.dim(1) == B.dim(1))
            nr = B.dim(2);
            nc = B.dim(3);
            nr_A = A.dim(2);
            nc_A = A.dim(3);
        } else {
            nr = B.dim(1);
            nc = B.dim(2);
            nr_A = A.dim(1);
            nc_A = A.dim(2);
        }

        NumR s1, s2;
        s1 = nr / (NumR) nr_A;
        s2 = nc / (NumR) nc_A;

        // get matrix
        MatR T_final, T;
        T_final.setSize(3, 3);
        math21_operator_mat_eye(T_final);

        math21_la_2d_matrix_scale(s1, s2, T);
        math21_operator_multiply_to_B(1, T, T_final);

        // transform
        math21_la_2d_affine_transform_image(A, B, T_final);
    }

    void math21_la_2d_matrix_test(const TenR &A, TenR &B) {
        // get matrix
        MatR T_final, T;
        T_final.setSize(3, 3);
        math21_operator_mat_eye(T_final);

        math21_la_2d_matrix_scale(0.5, 0.5, T);
        math21_operator_multiply_to_B(1, T, T_final);
        math21_la_2d_matrix_rotate(MATH21_PI / 6, T);
        math21_operator_multiply_to_B(1, T, T_final);
        math21_la_2d_matrix_shearing(0.5, 0.2, T);
        math21_operator_multiply_to_B(1, T, T_final);
        math21_la_2d_matrix_translate(50, 0, T);
        math21_operator_multiply_to_B(1, T, T_final);

        // transform
        math21_la_2d_affine_transform_image(A, B, T_final);
    }

    void math21_la_3d_matrix_translate(NumR x, NumR y, NumR z, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        T =
                1, 0, 0, x,
                0, 1, 0, y,
                0, 0, 1, z,
                0, 0, 0, 1;
    }

    void math21_la_3d_matrix_translate_reverse_mode(NumR x, NumR y, NumR z, MatR &T) {
        math21_la_3d_matrix_translate(-x, -y, -z, T);
    }

    void math21_la_3d_matrix_translate(const VecR &t, MatR &T) {
        MATH21_ASSERT(t.size() == 3);
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        T =
                1, 0, 0, t(1),
                0, 1, 0, t(2),
                0, 0, 1, t(3),
                0, 0, 0, 1;
    }

    // see math21_la_3d_set_KRt
    void math21_la_3d_matrix_K(NumR fx, NumR fy, NumR u0, NumR v0, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        T =
                fx, 0, u0, 0,
                0, fy, v0, 0,
                0, 0, 1, 1,
                0, 0, 1, 0;
    }

    // The meaning of R can be interpreted using inner product.
    // https://learnopengl.com/Advanced-OpenGL/Depth-testing
    // K' is chosen arbitrarily in order to be invertible. Or you can make it to have depth meaning.
    // K' = (K, c; c.t, 0) where c = (0, 0, 1).t
    // P' = K'(R, t; 0, 1), or P' = K'*T*R', where P' is 4*4, R' is 4*4 version of R, T is 4*4 version of t.
    // P = K(R, t) = (K, 0)*T*R' = removeRow3(P') where P is 3*4
    // return P'
    void math21_la_3d_matrix_KRt(const MatR &K, const MatR &mat_R, const MatR &_t, MatR &P) {
        MATH21_ASSERT(K.isSameSize(3, 3))
        MatR R, t;
        R = mat_R;
        t = _t;
        if (R.isEmpty()) {
            R.setSize(3, 3);
            math21_operator_mat_eye(R);
        }
        if (t.isEmpty()) {
            t.setSize(3);
            t = 0;
        }
        MatR K_prime(4, 4);
        K_prime =
                K(1, 1), K(1, 2), K(1, 3), 0,
                K(2, 1), K(2, 2), K(2, 3), 0,
                K(3, 1), K(3, 2), K(3, 3), 1,
                0, 0, 1, 0;
        MatR A(4, 4);
        if (R.isSameSize(3, 3)) {
            MATH21_ASSERT(t.size() == 3)
            A =
                    R(1, 1), R(1, 2), R(1, 3), t(1),
                    R(2, 1), R(2, 2), R(2, 3), t(2),
                    R(3, 1), R(3, 2), R(3, 3), t(3),
                    0, 0, 0, 1;
        } else {
            MATH21_ASSERT(R.isSameSize(4, 4))
            MATH21_ASSERT(t.isSameSize(4, 4))
            math21_op_mat_mul(t, R, A);
        }
        math21_op_mat_mul(K_prime, A, P);
    }

    // discard z axis.
    void math21_la_3x4_matrix_from_3d(const MatR &A, MatR &P) {
        P.setSize(3, 4);
        P =
                A(1, 1), A(1, 2), A(1, 3), A(1, 4),
                A(2, 1), A(2, 2), A(2, 3), A(2, 4),
                A(4, 1), A(4, 2), A(4, 3), A(4, 4);
    }

    void math21_la_3x4_matrix_KRt(const MatR &K, const MatR &R, const MatR &t, MatR &P) {
        MatR A;
        math21_la_3d_matrix_KRt(K, R, t, A);
        math21_la_3x4_matrix_from_3d(A, P);
    }

    void math21_la_3d_matrix_scale(NumR x, NumR y, NumR z, MatR &T) {
        MATH21_ASSERT(x != 0 && y != 0 && z != 0)
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        T =
                x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_matrix_scale_reverse_mode(NumR x, NumR y, NumR z, MatR &T) {
        MATH21_ASSERT(x != 0 && y != 0 && z != 0)
        math21_la_3d_matrix_scale(1 / x, 1 / y, 1 / z, T);
    }

    void math21_la_3d_matrix_rotate_about_x_axis(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = xjsin(theta);
        c = -xjsin(theta);
        d = xjcos(theta);
        T =
                1, 0, 0, 0,
                0, a, c, 0,
                0, b, d, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_matrix_rotate_about_y_axis(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = xjsin(theta);
        c = -xjsin(theta);
        d = xjcos(theta);
        T =
                d, 0, b, 0,
                0, 1, 0, 0,
                c, 0, a, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_matrix_rotate_about_z_axis(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = xjsin(theta);
        c = -xjsin(theta);
        d = xjcos(theta);
        T =
                a, c, 0, 0,
                b, d, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
    }

    //! returns 2x3 affine transformation matrix for the planar rotation.
    void getRotationMatrix2D(const VecR &x0, NumR theta, MatR &B) {
        MATH21_ASSERT(x0.nrows() == 2, "center is not in plane.");
        if (!B.isSameSize(2, 3)) {
            B.setSize(2, 3);
        }
        MatR b(2, 1), A(2, 2);
        A =
                xjcos(theta), -xjsin(theta),
                xjsin(theta), xjcos(theta);
        math21_operator_multiply(1, A, x0, b);
        math21_operator_linear_to_B(1.0, x0, -1.0, b);
//        b = x0 - A * x0;
        B =
                A(1, 1), A(1, 2), b(1, 1),
                A(2, 1), A(2, 2), b(2, 1);
    }

    // y = A(x-x0) + x0, so y_tilde = B * x_tilde
    void math21_la_getRotationMatrix2D(const VecR &x0, NumR theta, MatR &B) {
        MATH21_ASSERT(x0.nrows() == 2, "center is not in plane.");
        if (!B.isSameSize(3, 3)) {
            B.setSize(3, 3);
        }
        MatR b(2, 1), A(2, 2);
        A =
                xjcos(theta), -xjsin(theta),
                xjsin(theta), xjcos(theta);
        math21_operator_multiply(1, A, x0, b);
        math21_operator_linear_to_B(1.0, x0, -1.0, b);
//        b = x0 - A * x0;
        B =
                A(1, 1), A(1, 2), b(1, 1),
                A(2, 1), A(2, 2), b(2, 1),
                0, 0, 1;
    }

    void math21_la_rotate_and_translate_point(const VecR &b, NumR theta, MatR &T) {
        MATH21_ASSERT(b.nrows() == 2);
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        MatR A(2, 2);
        A =
                xjcos(theta), -xjsin(theta),
                xjsin(theta), xjcos(theta);
        T =
                A(1, 1), A(1, 2), b(1, 1),
                A(2, 1), A(2, 2), b(2, 1),
                0, 0, 1;
    }

    void math21_la_translate_and_rotate_axis_reverse_mode(const VecR &b, NumR theta, MatR &T) {
        math21_la_rotate_and_translate_point(b, theta, T);
    }

    void math21_la_rotate_axis(NumR theta, MatR &T) {
        if (!T.isSameSize(3, 3)) {
            T.setSize(3, 3);
        }
        MatR A(2, 2);
        A =
                xjcos(theta), xjsin(theta),
                -xjsin(theta), xjcos(theta);
        T =
                A(1, 1), A(1, 2), 0,
                A(2, 1), A(2, 2), 0,
                0, 0, 1;
    }

    void math21_la_3d_rotate_axis_about_x(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = -xjsin(theta);
        c = xjsin(theta);
        d = xjcos(theta);
        T =
                1, 0, 0, 0,
                0, a, c, 0,
                0, b, d, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_rotate_axis_about_y(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = -xjsin(theta);
        c = xjsin(theta);
        d = xjcos(theta);
        T =
                d, 0, b, 0,
                0, 1, 0, 0,
                c, 0, a, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_rotate_axis_about_z(NumR theta, MatR &T) {
        if (!T.isSameSize(4, 4)) {
            T.setSize(4, 4);
        }
        NumR a, b, c, d;
        a = xjcos(theta);
        b = -xjsin(theta);
        c = xjsin(theta);
        d = xjcos(theta);
        T =
                a, c, 0, 0,
                b, d, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
    }

    void math21_la_3d_post_rotate_axis_about_x(NumR theta, MatR &T) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        MatR R;
        math21_la_3d_rotate_axis_about_x(theta, R);
        math21_operator_multiply_to_B(1, R, T);
    }

    void math21_la_3d_post_rotate_axis_about_y(NumR theta, MatR &T) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        MatR R;
        math21_la_3d_rotate_axis_about_y(theta, R);
        math21_operator_multiply_to_B(1, R, T);
    }

    void math21_la_3d_post_rotate_axis_about_z(NumR theta, MatR &T) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        MatR R;
        math21_la_3d_rotate_axis_about_z(theta, R);
        math21_operator_multiply_to_B(1, R, T);
    }

    void math21_la_2d_affine_transform_image(const MatR &A, MatR &B, const MatR &T) {
        math21_la_affine_transform_image(A, B, T);
    }

    void math21_la_2d_affine_transform_image_reverse_mode(const MatR &A, MatR &B, const MatR &T) {
        math21_la_affine_transform_image_reverse_mode(A, B, T);
    }

    NumB math21_la_affine_transform_image(const MatR &A, MatR &B, const MatR &T) {
        MATH21_ASSERT(T.isSameSize(3, 3))
        MATH21_ASSERT(T(3, 1) == 0 && T(3, 2) == 0 && T(3, 3) == 1)
        MatR T_inv;
        if (!math21_operator_inverse(T, T_inv)) {
            return 0;
        }
        math21_la_affine_transform_image_reverse_mode(A, B, T_inv);
        return 1;
    }

    NumB math21_la_3d_perspective_projection_image(const MatR &A, MatR &B, const MatR &T) {
        MatR T_inv;
        if (!math21_operator_inverse(T, T_inv)) {
            return 0;
        }
        math21_la_3d_perspective_projection_image_reverse_mode(A, B, T_inv);

//        math21_la_3d_perspective_projection_image_normal_mode(A, B, T);
        return 1;
    }

    void math21_la_affine_transform_image_reverse_mode(const MatR &A, MatR &B, const MatR &T) {
        MATH21_ASSERT(T.isSameSize(3, 3))
        MATH21_ASSERT(math21_operator_num_isEqual(T(3, 1), 0, 1e-10)
                      && math21_operator_num_isEqual(T(3, 2), 0, 1e-10)
                      && math21_operator_num_isEqual(T(3, 3), 1, 1e-10), "" << T.log("T"))
        if (B.isEmpty()) {
            B.setSize(A.shape());
        }
        MATH21_ASSERT(A.dims() == B.dims())
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        NumN nch;
        NumN nr;
        NumN nc;
        NumN nr_A, nc_A;
        NumN dims = A.dims();
        if (A.dims() == 3) {
            MATH21_ASSERT(A.dim(1) == B.dim(1))
            nch = B.dim(1);
            nr = B.dim(2);
            nc = B.dim(3);
            nr_A = A.dim(2);
            nc_A = A.dim(3);
        } else {
            nr = B.dim(1);
            nc = B.dim(2);
            nch = 1;
            nr_A = A.dim(1);
            nc_A = A.dim(2);
        }

        NumN i1, i2, k;
        VecR x(3), y(3);
        x(3) = 1;
        NumN y1, y2;
        for (i1 = 1; i1 <= nr; ++i1) {
            for (i2 = 1; i2 <= nc; ++i2) {
                x(1) = i1;
                x(2) = i2;
                math21_operator_multiply(1, T, x, y);
                y1 = (NumN) y(1);
                y2 = (NumN) y(2);
                if (y1 >= 1 && y1 <= nr_A && y2 >= 1 && y2 <= nc_A) {
                    if (dims == 3) {
                        for (k = 1; k <= nch; ++k) {
                            B(k, i1, i2) = A(k, y1, y2);
                        }
                    } else {
                        B(i1, i2) = A(y1, y2);
                    }
                }
//                else {
//                    if (dims == 3) {
//                        for (k = 1; k <= nch; ++k) {
//                            B(k, i1, i2) = 0;
//                        }
//                    } else {
//                        B(i1, i2) = 0;
//                    }
//                }
            }
        }
    }

    void math21_la_3d_perspective_projection_image_normal_mode(const MatR &A, MatR &B, const MatR &T) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        if (B.isEmpty()) {
            B.setSize(A.shape());
        }
        MATH21_ASSERT(A.dims() == B.dims())
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        NumN nch;
        NumN nr;
        NumN nc;
        NumN nr_A, nc_A;
        NumN dims = A.dims();
        if (A.dims() == 3) {
            MATH21_ASSERT(A.dim(1) == B.dim(1))
            nch = B.dim(1);
            nr = B.dim(2);
            nc = B.dim(3);
            nr_A = A.dim(2);
            nc_A = A.dim(3);
        } else {
            nr = B.dim(1);
            nc = B.dim(2);
            nch = 1;
            nr_A = A.dim(1);
            nc_A = A.dim(2);
        }

        NumN i1, i2, k;
        VecR x(4), y(4);
        x(3) = 0;
        x(4) = 1;
        NumN y1, y2;
        for (i1 = 1; i1 <= nr_A; ++i1) {
            for (i2 = 1; i2 <= nc_A; ++i2) {
                x(1) = i1;
                x(2) = i2;
                math21_geometry_project_3d(T, x, y);
                y1 = (NumN) y(1);
                y2 = (NumN) y(2);
                if (y1 >= 1 && y1 <= nr && y2 >= 1 && y2 <= nc) {
                    if (dims == 3) {
                        for (k = 1; k <= nch; ++k) {
                            B(k, y1, y2) = A(k, i1, i2);
                        }
                    } else {
                        B(y1, y2) = A(i1, i2);
                    }
                }
            }
        }
    }

    // y = lambda * T * x, y = (y1, y2, 0, 1) => x3
    void math21_la_3d_perspective_projection_image_reverse_mode(const MatR &A, MatR &B, const MatR &T) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        MATH21_ASSERT(T(3, 3) != 0)
        if (B.isEmpty()) {
            B.setSize(A.shape());
        }
        MATH21_ASSERT(A.dims() == B.dims())
        MATH21_ASSERT(A.dims() == 3 || A.dims() == 2)
        NumN nch;
        NumN nr;
        NumN nc;
        NumN nr_A, nc_A;
        NumN dims = A.dims();
        if (A.dims() == 3) {
            MATH21_ASSERT(A.dim(1) == B.dim(1))
            nch = B.dim(1);
            nr = B.dim(2);
            nc = B.dim(3);
            nr_A = A.dim(2);
            nc_A = A.dim(3);
        } else {
            nr = B.dim(1);
            nc = B.dim(2);
            nch = 1;
            nr_A = A.dim(1);
            nc_A = A.dim(2);
        }

        NumN i1, i2, k;
        VecR x(4), y(4);
        x(3) = 0;
        x(4) = 1;
        NumN y1, y2;
        for (i1 = 1; i1 <= nr; ++i1) {
            for (i2 = 1; i2 <= nc; ++i2) {
                x(1) = i1;
                x(2) = i2;
                x(3) = (T(3, 1) * x(1) + T(3, 2) * x(2) + T(3, 4)) / (-T(3, 3));
                math21_geometry_project_3d(T, x, y);
                y1 = (NumN) y(1);
                y2 = (NumN) y(2);
                if (y1 >= 1 && y1 <= nr_A && y2 >= 1 && y2 <= nc_A) {
                    if (dims == 3) {
                        for (k = 1; k <= nch; ++k) {
                            B(k, i1, i2) = A(k, y1, y2);
                        }
                    } else {
                        B(i1, i2) = A(y1, y2);
                    }
                }
            }
        }
    }

    ////! R should be orthogonal, but here may not be.
    NumB geometry_cal_projection(const VecR &_x0, const VecR &_x1, const VecR &_x2, const VecR &_x3, MatR &T) {
        MATH21_ASSERT(_x0.size() == 3 && _x1.size() == 3 && _x2.size() == 3 && _x3.size() == 3)
        MATH21_ASSERT(!math21_operator_isEqualZero(_x0) && !math21_operator_isEqualZero(_x1)
                      && !math21_operator_isEqualZero(_x2) && !math21_operator_isEqualZero(_x3));
        // pre-process
        VecR x0, x1, x2, x3;
        x0.setSize(_x0.shape());
        x0.assign(_x0);
        x1.setSize(_x1.shape());
        x1.assign(_x1);
        x2.setSize(_x2.shape());
        x2.assign(_x2);
        x3.setSize(_x3.shape());
        x3.assign(_x3);
        if (x0(3) != 0) {
            math21_operator_linear_to(1.0 / x0(3), x0);
            x0(3) = 0;
        }
        if (x1(3) != 0) {
            math21_operator_linear_to(1.0 / x1(3), x1);
            math21_operator_linear_to_A(1.0, x1, -1.0, x0);
        }
        if (x2(3) != 0) {
            math21_operator_linear_to(1.0 / x2(3), x2);
            math21_operator_linear_to_A(1.0, x2, -1.0, x0);
        }
        if (x3(3) != 0) {
            math21_operator_linear_to(1.0 / x3(3), x3);
            math21_operator_linear_to_A(1.0, x3, -1.0, x0);
        }
        NumR num = x1(3) + x2(3) + x3(3);
        if (num < 2.0) {
            m21error("no focus");
            return 0;
        }

        // calculate f
        NumR f12 = 0.0, f13 = 0.0, f23 = 0.0, f;
        NumR tmp;
        if (x1(3) != 0 && x2(3) != 0) {
            tmp = math21_operator_InnerProduct(1.0, x1, x2);
            tmp = tmp - 1;
            tmp = -tmp;
            if (tmp < 0) {// we give it a chance because we think it is accidently negative.
//                m21log("f12 negative");
                x2(1) = x2(1) * (-1);
                x2(2) = x2(2) * (-1);
                f12 = xjsqrt(-tmp);
            } else if (tmp > 0) {
                f12 = xjsqrt(tmp);
            }
        }
        if (x1(3) != 0 && x3(3) != 0) {
            tmp = math21_operator_InnerProduct(1.0, x1, x3);
            tmp = tmp - 1;
            tmp = -tmp;
            if (tmp > 0) {
                f13 = xjsqrt(tmp);
            }
        }
        if (x2(3) != 0 && x3(3) != 0) {
            tmp = math21_operator_InnerProduct(1.0, x2, x3);
            tmp = tmp - 1;
            tmp = -tmp;
            if (tmp > 0) {
                f23 = xjsqrt(tmp);
            }
        }
        m21log("\nf12", f12);
        m21log("f13", f13);
        m21log("f23", f23);
        f = xjmax(f12, f13);
        f = xjmax(f, f23);
        if (f < MATH21_EPS) {
            m21error("no focus");
            return 0;
        }
        if (x1(3) != 0) {
            x1(3) = f;
        }
        if (x2(3) != 0) {
            x2(3) = f;
        }
        if (x3(3) != 0) {
            x3(3) = f;
        }

        math21_operator_linear_to(1 / math21_operator_norm(x1, 2), x1);
        math21_operator_linear_to(1 / math21_operator_norm(x2, 2), x2);
        math21_operator_linear_to(1 / math21_operator_norm(x3, 2), x3);

        // compute T
        MatR K_inv(3, 3), R(3, 3);

        K_inv =
                1 / f, 0, -x0(1) / f,
                0, 1 / f, -x0(2) / f,
                0, 0, 1;
        R =
                x1(1), x2(1), x3(1),
                x1(2), x2(2), x3(2),
                x1(3), x2(3), x3(3);

        if (!math21_operator_inverse(R)) {
            return 0;
        }
        math21_operator_multiply(1.0, R, K_inv, T);
        return 1;
    }

    // compute s, T = S*T. Here we assume s = sx = sy.
    NumN geometry_cal_projection_scale(const VecR &xa, const VecR &xb, NumR l, MatR &T) {
        MATH21_ASSERT(xa.size() == 3 && xb.size() == 3)

        VecR Xa, Xb, Xab;
        NumN flag;
        flag = math21_geometry_project(T, xa, Xa);
        if (flag == 0) {
            m21error("project xa");
            return 0;
        }
        flag = math21_geometry_project(T, xb, Xb);
        if (flag == 0) {
            m21error("project xb");
            return 0;
        }

        math21_operator_linear(1.0, Xa, -1.0, Xb, Xab);
        if (math21_operator_norm(Xab, 2) < MATH21_EPS) {
            m21error("Xab.norm(2) too small");
            return 0;
        }
        NumR s = l / math21_operator_norm(Xab, 2);
        if (s < MATH21_EPS) {
            m21error("s too small");
            return 0;
        }
//        m21log("s", s);
//        T.log("T");
        MatR S;
        S.setSize(3, 3);
        S =
                s, 0, 0,
                0, s, 0,
                0, 0, 1;
        MatR T_tmp;
        math21_operator_multiply(1.0, S, T, T_tmp);
        T.assign(T_tmp);
        return 1;
    }

    // compute sx, sy, T = S*T. Here just get A, b.
    NumB geometry_cal_projection_scale_subroutine_01(const Seqce<VecR> &xas, const Seqce<VecR> &xbs,
                                                     const Seqce<NumR> &ls, const MatR &T,
                                                     MatR &A, VecR &b, VecR &ls_predict) {
        NumN N = xas.size();
        MATH21_ASSERT(N == 2 && xbs.size() == N && ls.size() == N);
        for (NumN i = 1; i <= N; ++i) {
            MATH21_ASSERT(xas(i).size() == 3 && xbs(i).size() == 3)
        }

        if (A.nrows() != N || A.ncols() != 2) {
            A.setSize(N, 2);
        }
        if (b.size() != N) {
            b.setSize(N);
        }
        if (ls_predict.size() != N) {
            ls_predict.setSize(N);
        }
        VecR Xab;
        NumN flag;

        for (NumN i = 1; i <= N; ++i) {
            const VecR &xa = xas(i);
            const VecR &xb = xbs(i);
            const NumR &l = ls(i);

            flag = math21_geometry_project(T, xa, xb, Xab);
            if (flag == 0) {
                m21error("project xa, xb to Xab");
                return 0;
            }
            if (math21_operator_norm(Xab, 2) < MATH21_EPS) {
                m21error("Xab.norm(2) too small");
                return 0;
            }
            A(i, 1) = xjsquare(Xab(1));
            A(i, 2) = xjsquare(Xab(2));
            b(i) = xjsquare(l);
            ls_predict(i) = math21_operator_norm(Xab, 2);
        }
//        ls_predict.log("ls_predict");
        return 1;
    }

    // compute sx, sy, T = S*T.
    NumB geometry_cal_projection_scale_subroutine_02(MatR &A, VecR &b, MatR &T) {
        MatR S;
        S.setSize(3, 3);

//        A.log("A");
        VecR x;
        if (!math21_operator_solve_linear_equation(A, b, x)) {
            return 0;
        }

        if (x(1) < MATH21_EPS || x(2) < MATH21_EPS) {
            m21error("solution x of linear equation not positive");
            A.log("A");
            b.log("b");
            x.log("x");
            return 0;
        }
        NumR sx = xjsqrt(x(1));
        NumR sy = xjsqrt(x(2));
//        m21log("sx", sx);
//        m21log("sy", sy);
//        T.log("T");
        S =
                sx, 0, 0,
                0, sy, 0,
                0, 0, 1;
        MatR T_tmp;
        math21_operator_multiply(1.0, S, T, T_tmp);
        T.assign(T_tmp);
        return 1;
    }

    // compute sx, sy, T = S*T.
    NumB geometry_cal_projection_scale(const Seqce<VecR> &xas, const Seqce<VecR> &xbs,
                                       const Seqce<NumR> &ls, MatR &T) {
        NumN N = xas.size();
        MATH21_ASSERT(N == 2 && xbs.size() == N && ls.size() == N);
        for (NumN i = 1; i <= N; ++i) {
            MATH21_ASSERT(xas(i).size() == 3 && xbs(i).size() == 3)
        }

        MatR A;
        VecR b;
        VecR ls_predict;

        NumN flag;

        flag = geometry_cal_projection_scale_subroutine_01(xas, xbs,
                                                           ls, T,
                                                           A, b, ls_predict);
        if (flag == 0) {
            return 0;
        }

        flag = geometry_cal_projection_scale_subroutine_02(A, b, T);
        if (flag == 0) {
            return 0;
        }

        flag = geometry_cal_projection_scale_subroutine_01(xas, xbs,
                                                           ls, T,
                                                           A, b, ls_predict);
        if (flag == 0) {
            return 0;
        }
        return 1;
    }

    // X = T*x, xs * T_trans = Xs
    NumB math21_geometry_cal_affine(const MatR &xs, const MatR &Xs, MatR &T) {
        MATH21_ASSERT(xs.isSameSize(3, 3));
        MATH21_ASSERT(Xs.isSameSize(3, 3));
        if (!math21_operator_solve_linear_equation(xs, Xs, T)) {
            return 0;
        }
        math21_operator_matrix_trans(T);
//        T.log("affine T");
        MATH21_ASSERT(math21_operator_num_isEqual(T(3, 1), 0, 1e-10));
        MATH21_ASSERT(math21_operator_num_isEqual(T(3, 2), 0, 1e-10));
        MATH21_ASSERT(math21_operator_num_isEqual(T(3, 3), 1, 1e-10));
        return 1;
    }

    // diag = {(x1,y1), (x2, y2)} as diagonal.
    // X = T * x, x in rectangle with diagonal diag, X in image(nr, nc).
    NumB
    math21_geometry_cal_affine_matrix_axis_to_matrix(MatR &T, NumR x1, NumR x2, NumR y1, NumR y2, NumN nr, NumN nc) {
        NumR y_min, y_max, x_min, x_max;
        x_min = xjmin(x1, x2);
        y_min = xjmin(y1, y2);
        x_max = xjmax(x1, x2);
        y_max = xjmax(y1, y2);
        MatR xs(3, 3);
        MatR Xs(3, 3);
        xs =
                x_min, y_min, 1,
                x_max, y_min, 1,
                x_min, y_max, 1;
        Xs =
                nr, 1, 1,
                nr, nc, 1,
                1, 1, 1;
        return math21_geometry_cal_affine(xs, Xs, T);
    }

    // compute affine matrix W, T = W*T.
    NumB geometry_cal_projection_to_world(const MatR &Xs, const MatR &xs, MatR &T) {
        MATH21_ASSERT(Xs.isSameSize(3, 3));
        MATH21_ASSERT(xs.isSameSize(3, 3));
        for (NumN i = 1; i <= 3; ++i) {
            MATH21_ASSERT(Xs(i, 3) == 1 && xs(i, 3) == 1)
        }

        MatR Xs_tilde;
        MatR W;
        NumB flag;
        flag = math21_geometry_project_in_batch(T, xs, Xs_tilde);
        if (flag == 0) {
            m21error("project xs");
            return 0;
        }

        // Xs_tilde * W_trans = Xs.
        math21_geometry_cal_affine(Xs_tilde, Xs, W);
        math21_operator_multiply_to_B(1.0, W, T);
        return 1;
    }

    NumB math21_geometry_points_homogeneous_to_non(const MatR &X, MatR &Y, NumB ignoreBad) {
        NumN nr = X.nrows();
        NumN nc = X.ncols();
        math21_tool_assert(nc > 1);
        Y.setSize(nr, nc - 1);
        for (NumN i = 1; i <= nr; ++i) {
            if (xjabs(X(i, nc)) < MATH21_EPS) {
                if (!ignoreBad) {
                    return 0;
                } else {
                    for (NumN j = 1; j <= nc - 1; ++j) {
                        Y(i, j) = 0;
                    }
                }
            } else {
                for (NumN j = 1; j <= nc - 1; ++j) {
                    Y(i, j) = X(i, j) / X(i, nc);
                }
            }
        }
        return 1;
    }

    // see math21_device_geometry_project_2d
    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*3.
    NumN math21_geometry_project(const MatR &T, const VecR &x, VecR &X) {
        math21_operator_multiply(1.0, T, x, X);
        if (xjabs(X(3)) < MATH21_EPS) {
            return 0;
        }
        math21_operator_linear_to(1.0 / X(3), X);
        return 1;
    }

    // see math21_geometry_project
    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*3.
    NumB math21_device_geometry_project_2d(const NumR *T, const NumR *x, NumR *X) {
        NumN i, j;
        NumR sum;
        for (i = 1; i <= 3; ++i) {
            sum = 0;
            for (j = 1; j <= 3; ++j) {
                sum += T[(i - 1) * 3 + j] * x[j];
            }
            X[i] = sum;
        }
        if (math21_device_f_abs(X[3]) < MATH21_EPS) {
            return 0;
        }
        for (i = 1; i <= 3; ++i) {
            X[i] /= X[3];
        }
        return 1;
    }

    // x = K * X, where x.t = (u,v,1), X.t = (xc, yc, 1), K = (fx, 0, u0; 0, fy, v0; 0, 0, 1)
    void math21_device_geometry_K_project_2d(NumR *u, NumR *v, NumR xc, NumR yc,
                                             NumR fx, NumR fy, NumR u0, NumR v0) {
        *u = fx * xc + u0;
        *v = fy * yc + v0;
    }

    // X = K.inv * x, where x.t = (u,v,1), X.t = (xc, yc, 1), K = (fx, 0, u0; 0, fy, v0; 0, 0, 1)
    void math21_device_geometry_K_inverse_project_2d(NumR u, NumR v, NumR *xc, NumR *yc,
                                                     NumR fx, NumR fy, NumR u0, NumR v0) {
        *xc = (u - u0) / fx;
        *yc = (v - v0) / fy;
    }

    // X = lambda * T * x; lambda is set to make sure X(4) = 1. Here T is 4*4.
    NumB math21_geometry_project_3d(const MatR &T, const VecR &x, VecR &X) {
        MATH21_ASSERT(T.isSameSize(4, 4))
        math21_operator_multiply(1.0, T, x, X);
        if (xjabs(X(4)) < MATH21_EPS) {
            return 0;
        }
        math21_operator_linear_to(1.0 / X(4), X);
        return 1;
    }

    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*4.
    NumB math21_geometry_project_3x4(const MatR &T, const VecR &x, VecR &X) {
        MATH21_ASSERT(T.isSameSize(3, 4))
        math21_operator_multiply(1.0, T, x, X);
        if (xjabs(X(3)) < MATH21_EPS) {
            return 0;
        }
        math21_operator_linear_to(1.0 / X(3), X);
        return 1;
    }

    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*3.
    NumB math21_geometry_project_2d_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs) {
        MATH21_ASSERT(xs.dims() == 2 && xs.dim(2) == 2)
        if (!Xs.isSameSize(xs.nrows(), 2)) {
            Xs.setSize(xs.nrows(), 2);
        }

        NumB flag;
        VecR x(3), X(3);
        for (NumN i = 1; i <= xs.nrows(); ++i) {
            x = xs(i, 1), xs(i, 2), 1;
            flag = math21_geometry_project(T, x, X);
            if (!flag) {
                return 0;
            }
            Xs(i, 1) = X(1);
            Xs(i, 2) = X(2);
        }
        return 1;
    }

    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*3.
    NumB math21_geometry_project_in_batch(const MatR &T, const MatR &xs, MatR &Xs) {
        math21_operator_multiply_trans(1, xs, T, Xs);

        NumR ratio;
        for (NumN i = 1; i <= Xs.nrows(); ++i) {
            if (xjabs(Xs(i, 3)) < MATH21_EPS) {
                return 0;
            }
            ratio = Xs(i, 3);
            Xs(i, 1) /= ratio;
            Xs(i, 2) /= ratio;
            Xs(i, 3) = 1;
        }
        return 1;
    }

    // X = lambda * T * x; lambda is set to make sure X(4) = 1. Here T is 4*4.
    NumB math21_geometry_project_3d_in_batch(const MatR &T, const MatR &xs, MatR &Xs) {
        math21_operator_multiply_trans(1, xs, T, Xs);

        NumR ratio;
        for (NumN i = 1; i <= Xs.nrows(); ++i) {
            if (xjabs(Xs(i, 4)) < MATH21_EPS) {
                return 0;
            }
            ratio = Xs(i, 4);
            Xs(i, 1) /= ratio;
            Xs(i, 2) /= ratio;
            Xs(i, 3) /= ratio;
            Xs(i, 4) = 1;
        }
        return 1;
    }

    // X = lambda * T * x; lambda is set to make sure X(4) = 1. Here T is 4*4.
    NumB math21_geometry_project_3d_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs) {
        MATH21_ASSERT(xs.dims() == 2 && xs.dim(2) == 3)
        if (!Xs.isSameSize(xs.nrows(), 3)) {
            Xs.setSize(xs.nrows(), 3);
        }

        NumB flag;
        VecR x(4), X(4);
        for (NumN i = 1; i <= xs.nrows(); ++i) {
            x = xs(i, 1), xs(i, 2), xs(i, 3), 1;
            flag = math21_geometry_project_3d(T, x, X);
            if (!flag) {
                return 0;
            }
            Xs(i, 1) = X(1);
            Xs(i, 2) = X(2);
            Xs(i, 3) = X(3);
        }
        return 1;
    }

    // see math21_geometry_project_3d_non_homogeneous_in_batch
    // X = lambda * T * x; lambda is set to make sure X(3) = 1. Here T is 3*4.
    NumB math21_geometry_project_3x4_non_homogeneous_in_batch(const MatR &T, const MatR &xs, MatR &Xs) {
        MATH21_ASSERT(xs.dims() == 2 && xs.dim(2) == 3)
        if (!Xs.isSameSize(xs.nrows(), 2)) {
            Xs.setSize(xs.nrows(), 2);
        }

        NumB flag;
        VecR x(4), X(3);
        for (NumN i = 1; i <= xs.nrows(); ++i) {
            x = xs(i, 1), xs(i, 2), xs(i, 3), 1;
            flag = math21_geometry_project_3x4(T, x, X);
            if (!flag) {
                return 0;
            }
            Xs(i, 1) = X(1);
            Xs(i, 2) = X(2);
        }
        return 1;
    }

    NumB math21_geometry_project_Interval2D(const MatR &T, const Interval2D &I, MatR &vertices) {
        MatR xs(4, 3);
        NumR a1 = I(1).left();
        NumR b1 = I(1).right();
        NumR a2 = I(2).left();
        NumR b2 = I(2).right();
        xs =
                a1, a2, 1,
                b1, a2, 1,
                b1, b2, 1,
                a1, b2, 1;

        MatR Xs;
        NumB flag;
        flag = math21_geometry_project_in_batch(T, xs, Xs);
        if (!flag) {
            return 0;
        }
        if (!vertices.isSameSize(4, 2)) {
            vertices.setSize(4, 2);
        }
        math21_operator_matrix_delete_col(Xs, vertices, 3);
        return 1;
    }

    // Xab = Xb - Xa. Here T is 3*3.
    NumN math21_geometry_project(const MatR &T, const VecR &xa, const VecR &xb, VecR &Xab) {

        NumN flag;
        VecR Xa, Xb;
        flag = math21_geometry_project(T, xa, Xa);
        if (flag == 0) {
            m21error("project xa");
            return 0;
        }
        flag = math21_geometry_project(T, xb, Xb);
        if (flag == 0) {
            m21error("project xb");
            return 0;
        }
        math21_operator_linear(1.0, Xa, -1.0, Xb, Xab);
        return 1;
    }

    // index(1), index(2) keeps the X(1), X(2) respectively.
    // X = lambda * T * x; x is in image(nr, nc);
    NumB math21_geometry_project_cal_image_index(const MatR &T, NumN nr, NumN nc, TenR &index) {
        if (!index.isSameSize(2, nr, nc)) {
            index.setSize(2, nr, nc);
        }

        VecR x(3), X(3);
        for (NumN i = 1; i <= nr; ++i) {
            for (NumN j = 1; j <= nc; ++j) {
                x = i, j, 1;
                math21_geometry_project(T, x, X);
                index(1, i, j) = X(1);
                index(2, i, j) = X(2);
            }
        }
        return 1;
    }

    // A is camera matrix.
    // nr, nc are resolution of camera frame.
    void math21_geometry_project_cal_camera_characteristics(const MatR &A, NumN nr, NumN nc,
                                                            NumR &fovx, NumR &fovy,
                                                            NumR &aspectRatio) {
        MATH21_ASSERT(A.isSameSize(3, 3))

        /* Calculate pixel aspect ratio. */
        aspectRatio = A(2, 2) / A(1, 1);

        /* Calculate fovx and fovy. */
        fovx = xjatan2(A(1, 3), A(1, 1)) + atan2(nr - A(1, 3), A(1, 1));
        fovy = xjatan2(A(2, 3), A(2, 2)) + atan2(nc - A(2, 3), A(2, 2));
        fovx *= 180.0 / XJ_PI;
        fovy *= 180.0 / XJ_PI;
    }

    void math21_geometry_project_matrix_to_standard(const MatR &A,
                                                    const VecR &mat_distortion,
                                                    const MatR &mat_R,
                                                    const MatR &mat_A_new,
                                                    VecR &distortion,
                                                    MatR &R,
                                                    MatR &A_new) {
        MATH21_ASSERT(A.isSameSize(3, 3))
        MATH21_ASSERT(mat_A_new.isEmpty() || mat_A_new.isSameSize(3, 3) || mat_A_new.isSameSize(3, 4),
                      "" << mat_A_new.logInfo("mat_A_new"));
        MATH21_ASSERT(mat_R.isEmpty() || mat_R.isSameSize(3, 3))
        MATH21_ASSERT(mat_distortion.isEmpty()
                      || mat_distortion.isSameSize(4) || mat_distortion.isSameSize(4, 1)
                      || mat_distortion.isSameSize(5) || mat_distortion.isSameSize(5, 1)
                      || mat_distortion.isSameSize(8) || mat_distortion.isSameSize(8, 1)
                      || mat_distortion.isSameSize(12) || mat_distortion.isSameSize(12, 1)
                      || mat_distortion.isSameSize(14) || mat_distortion.isSameSize(14, 1),
                      "" << mat_distortion.logInfo("mat_distortion"))

        if (mat_A_new.isEmpty()) {
            A_new = A;
        } else {
            A_new.setSize(A.shape());
            math21_op_matrix_sub_region_tl_set(mat_A_new, A_new);
        }

        R.setSize(3, 3);
        if (mat_R.isEmpty()) {
            math21_operator_mat_eye(R);
        } else {
            R = mat_R;
        }

        distortion.setSize(8); // ignore i-th coefficient with i > 8.
        distortion = 0;
        if (!mat_distortion.isEmpty()) {
            math21_op_matrix_sub_region_tl_set(mat_distortion, distortion);
        }
    }

    // nr, nc are of distorted image.
    // A is camera matrix of distorted image.
    // distortion is distortion coefficients of distorted image.
    // R is rotation matrix of corrected image.
    // A_new is camera matrix of corrected image.
    // inner is inner rectangle of corrected image.
    // outer is outer rectangle of corrected image.
    NumB math21_geometry_project_perspective_get_rectangles_from_corrected(NumN nr, NumN nc, const MatR &A,
                                                                           const VecR &mat_distortion,
                                                                           const MatR &mat_R,
                                                                           const MatR &mat_A_new,
                                                                           Interval2D &inner,
                                                                           Interval2D &outer,
                                                                           NumB isUseInner,
                                                                           NumB isUseOuter,
                                                                           NumB isNormalized) {
        NumB flag;
        const NumN N = 9;
        MatR ps(N * N, 2);
        NumN k = 1;
        NumR r1 = (nr - 1) / (NumR) (N - 1);
        NumR r2 = (nc - 1) / (NumR) (N - 1);
        for (NumN i = 0; i < N; ++i) {
            for (NumN j = 0; j < N; ++j) {
                ps(k, 1) = 1 + i * r1;
                ps(k, 2) = 1 + j * r2;
                ++k;
            }
        }

        MatR ps_new;
        flag = math21_geometry_project_perspective_points_distorted_to_corrected(ps, ps_new, A, mat_distortion, mat_R,
                                                                                 mat_A_new,
                                                                                 isNormalized);
        if (!flag) {
            return 0;
        }

        NumR iX0 = -NumR_MAX, iX1 = NumR_MAX, iY0 = -NumR_MAX, iY1 = NumR_MAX;
        NumR oX0 = NumR_MAX, oX1 = -NumR_MAX, oY0 = NumR_MAX, oY1 = -NumR_MAX;
        // find the inscribed rectangle.
        k = 1;
        for (NumN i = 1; i <= N; ++i) {
            for (NumN j = 1; j <= N; ++j) {
                NumR x = ps_new(k, 1);
                NumR y = ps_new(k, 2);
                oX0 = xjmin(oX0, x);
                oX1 = xjmax(oX1, x);
                oY0 = xjmin(oY0, y);
                oY1 = xjmax(oY1, y);

                if (i == 1)
                    iX0 = xjmax(iX0, x);
                if (i == N)
                    iX1 = xjmin(iX1, x);
                if (j == 1)
                    iY0 = xjmax(iY0, y);
                if (j == N)
                    iY1 = xjmin(iY1, y);
                ++k;
            }
        }
        if (isUseInner) {
            inner.at(1).set(iX0, iX1, 1, 1);
            inner.at(2).set(iY0, iY1, 1, 1);
        }
        if (isUseOuter) {
            outer.at(1).set(oX0, oX1, 1, 1);
            outer.at(2).set(oY0, oY1, 1, 1);
        }
        return 1;
    }

    // not inner actually. Need improvement for circle shape.
    NumB math21_geometry_project_fisheye_get_rectangles_from_corrected(NumN nr, NumN nc, const MatR &A,
                                                                       const VecR &mat_distortion,
                                                                       const MatR &mat_R,
                                                                       const MatR &mat_A_new,
                                                                       VecR &center_mass,
                                                                       Interval2D &inner,
                                                                       Interval2D &outer,
                                                                       NumB isUseInner,
                                                                       NumB isUseOuter,
                                                                       NumB isNormalized) {
        MATH21_ASSERT(isUseOuter == 0, "outer rectangle unsupported!")
        NumB flag;
        const NumN N = 4;
        MatR ps(N, 2);
        ps =
                1, (nc + 1) / 2.0,
                (nr + 1) / 2.0, nc,
                nr, (nc + 1) / 2.0,
                (nr + 1) / 2.0, 1;

        MatR ps_new;
        flag = math21_geometry_project_fisheye_points_distorted_to_corrected(ps, ps_new, A, mat_distortion, mat_R,
                                                                             mat_A_new,
                                                                             isNormalized);
        if (!flag) {
            return 0;
        }

        math21_operator_matrix_col_mean(ps_new, center_mass);

        NumR minx = NumR_MAX, miny = NumR_MAX, maxx = -NumR_MAX, maxy = -NumR_MAX;
        for (NumN i = 1; i <= N; ++i) {
            minx = xjmin(minx, ps_new(i, 1));
            maxx = xjmax(maxx, ps_new(i, 1));
            miny = xjmin(miny, ps_new(i, 2));
            maxy = xjmax(maxy, ps_new(i, 2));
        }

        if (isUseInner) {
            inner.at(1).set(minx, maxx, 1, 1);
            inner.at(2).set(miny, maxy, 1, 1);
        }
        return 1;
    }

    NumB math21_geometry_project_fisheye_points_distorted_to_corrected(const MatR &src, MatR &dst, const MatR &A,
                                                                       const VecR &mat_distortion,
                                                                       const MatR &mat_R,
                                                                       const MatR &mat_A_new,
                                                                       NumB isNormalized) {
        MATH21_ASSERT(A.isSameSize(3, 3))
        MATH21_ASSERT(mat_A_new.isEmpty() || mat_A_new.isSameSize(3, 3))
        MATH21_ASSERT(mat_R.isEmpty() || mat_R.isSameSize(3, 3))
        MATH21_ASSERT(mat_distortion.isEmpty()
                      || mat_distortion.isSameSize(4) || mat_distortion.isSameSize(4, 1))
        MATH21_ASSERT(!src.isEmpty() && src.ncols() == 2)

        if (!dst.isSameSize(src.shape())) {
            dst.setSize(src.shape());
        }

        MatR A_new;
        if (mat_A_new.isEmpty()) {
            A_new.setSize(A.shape());
            A_new = A;
        } else {
            A_new.setSize(mat_A_new.shape());
            A_new = mat_A_new;
        }
        if (isNormalized) {
            math21_operator_mat_eye(A_new);
        }

        MatR R;
        R.setSize(3, 3);
        if (mat_R.isEmpty()) {
            math21_operator_mat_eye(R);
        } else {
            R = mat_R;
        }

        math21_operator_multiply_to_B(1.0, A_new, R);

        NumN iters = 0;
        VecR distortion;
        distortion.setSize(4);
        distortion = 0;
        if (!mat_distortion.isEmpty()) {
            math21_operator_container_set_partially(mat_distortion, distortion, 0, 0, mat_distortion.size());
            iters = 10;
        }

        NumR u0 = A(1, 3), v0 = A(2, 3);
        NumR fx = A(1, 1), fy = A(2, 2);
        NumR ifx = 1. / fx, ify = 1. / fy;

        NumR k1, k2, k3, k4;

        k1 = distortion(1);
        k2 = distortion(2);
        k3 = distortion(3);
        k4 = distortion(4);

        for (NumN i = 1; i <= src.nrows(); ++i) {
            NumR x, y;
            x = src(i, 1);
            y = src(i, 2);
            x = (x - u0) * ifx;
            y = (y - v0) * ify;

            NumR iscale = 1.0;

            NumR theta_d = xjsqrt(x * x + y * y);

            // the current camera model is only valid up to 180 FOV
            theta_d = xjmin(xjmax(-XJ_PI / 2., theta_d), XJ_PI / 2.);

            if (theta_d > 1e-8 && iters) {
                // compensate distortion iteratively
                NumR theta = theta_d;

                const NumR EPS = 1e-8;
                for (NumN j = 1; j <= iters; j++) {
                    NumR theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                            theta6 * theta2;
                    NumR k0_theta2 = k1 * theta2, k1_theta4 = k2 * theta4, k2_theta6 = k3 * theta6, k3_theta8 =
                            k4 * theta8;
                    NumR theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                     (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
                    theta = theta - theta_fix;
                    if (fabs(theta_fix) < EPS)
                        break;
                }

                NumR r = xjtan(theta);
                iscale = r / theta_d;
            }
            x = x * iscale;
            y = y * iscale;

            NumR xx = R(1, 1) * x + R(1, 2) * y + R(1, 3);
            NumR yy = R(2, 1) * x + R(2, 2) * y + R(2, 3);
            NumR ww = 1. / (R(3, 1) * x + R(3, 2) * y + R(3, 3));
            x = xx * ww;
            y = yy * ww;
            dst(i, 1) = x;
            dst(i, 2) = y;
        }
        return 1;
    }

    // see cv::undistortPoints
    // src is {*,*; *,*; *,*} in distorted image.
    // dst is {*,*; *,*; *,*} in corrected image.
    // If it is normalized, A_new will be omitted, and dst will contain normalized point coordinates.
    // A is camera matrix of distorted image.
    // distortion is distortion coefficients of distorted image.
    // R is rotation matrix of corrected image.
    // A_new is camera matrix of corrected image.
    // ki are radial distortion coefficients.
    // pi are tangential distortion coefficients.
    // no thin prism distortion.
    // no trapezoidal distortion.
    // barrel distortion: k1<0.
    // pincushion distortion: k1>0.
    NumB math21_geometry_project_perspective_points_distorted_to_corrected(
            const MatR &src, MatR &dst, const MatR &A,
            const VecR &mat_distortion,
            const MatR &mat_R,
            const MatR &mat_A_new,
            NumB isNormalized) {
        if (src.isEmpty())return 0;
        MatR R, distortion, A_new;
        math21_geometry_project_matrix_to_standard(A, mat_distortion, mat_R, mat_A_new, distortion, R, A_new);
        MATH21_ASSERT(src.ncols() == 2)
        if (!dst.isSameSize(src.shape())) {
            dst.setSize(src.shape());
        }

        if (isNormalized) {
            math21_operator_mat_eye(A_new);
        }
        math21_operator_multiply_to_B(1.0, A_new, R);

        NumN iters = 0;
        if (!mat_distortion.isEmpty()) {
            iters = 6;
        }

        NumR u0 = A(1, 3), v0 = A(2, 3);
        NumR fx = A(1, 1), fy = A(2, 2);
        NumR ifx = 1. / fx, ify = 1. / fy;

        NumR k1, k2, p1, p2, k3, k4, k5, k6;

        k1 = distortion(1);
        k2 = distortion(2);
        p1 = distortion(3);
        p2 = distortion(4);
        k3 = distortion(5);
        k4 = distortion(6);
        k5 = distortion(7);
        k6 = distortion(8);

        VecR p(3), P(3);
        for (NumN i = 1; i <= src.nrows(); ++i) {
            NumR x, y, x0 = 0, y0 = 0;
            x = src(i, 1);
            y = src(i, 2);

            x = (x - u0) * ifx;
            y = (y - v0) * ify;
            if (iters) { // compensate distortion iteratively
                x0 = x;
                y0 = y;
                for (NumN j = 1; j <= iters; j++) {
                    NumR r2 = x * x + y * y;
                    NumR ikr = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
                    NumR deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
                    NumR deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
                    x = (x0 - deltaX) * ikr;
                    y = (y0 - deltaY) * ikr;
                }
            }
            NumR xx = R(1, 1) * x + R(1, 2) * y + R(1, 3);
            NumR yy = R(2, 1) * x + R(2, 2) * y + R(2, 3);
            NumR ww = 1. / (R(3, 1) * x + R(3, 2) * y + R(3, 3));
            x = xx * ww;
            y = yy * ww;
            dst(i, 1) = x;
            dst(i, 2) = y;
        }
        return 1;
    }

    // Bad for circle.
    NumB math21_geometry_project_fisheye_get_camera_matrix_combined_with_viewport(const MatR &A,
                                                                                  const VecR &mat_distortion,
                                                                                  NumN nr_old, NumN nc_old,
                                                                                  MatR &mat_A_new,
                                                                                  NumN nr_new, NumN nc_new,
                                                                                  NumR alpha) {
        NumB flag;
        if (nr_new * nc_new == 0) {
            nr_new = nr_old;
            nc_new = nc_old;
        }

        MatR &A_new = mat_A_new;
        if (!A_new.isSameSize(A.shape())) {
            A_new.setSize(A.shape());
        }

        Interval2D inner;
        Interval2D outer;
        NumB isUseInner = 1, isUseOuter = 1;

        // Todo: two kinds of methods, but the normalized needs proof.
        NumB isNormalized = 1;
//        NumB isNormalized = 0;
        VecR center_mass;
        flag = math21_geometry_project_fisheye_get_rectangles_from_corrected(nr_old, nc_old, A, mat_distortion,
                                                                             MatR(), MatR(), center_mass, inner, outer,
                                                                             isUseInner,
                                                                             0, isNormalized);
        if (!flag) {
            return 0;
        }
        NumR f1 = nr_new * 0.5 / (center_mass(1) - inner(1).left());
        NumR f2 = nr_new * 0.5 / (inner(1).right() - center_mass(1));
        NumR f3 = nc_new * 0.5 / (center_mass(2) - inner(2).left());
        NumR f4 = nc_new * 0.5 / (inner(2).right() - center_mass(2));

        NumR fmin = xjmin(f1, xjmin(f2, xjmin(f3, f4)));
        NumR fmax = xjmax(f1, xjmax(f2, xjmax(f3, f4)));

        NumR f = alpha * fmin + (1.0 - alpha) * fmax;

        NumR fx, fy, cx, cy;
        fx = f;
        fy = f;

        cx = -center_mass(1) * f + nr_new * 0.5;
        cy = -center_mass(2) * f + nc_new * 0.5;

        NumR aspect_ratio;
        aspect_ratio = A(1, 1) / A(2, 2);

        // restore aspect ratio
        fy = fy / aspect_ratio;

        MatR viewport(3, 3);
        viewport =
                fx, 0, cx,
                0, fy, cy,
                0, 0, 1;
        if (isNormalized) {
            A_new = viewport;
        } else {
            math21_operator_multiply(1.0, viewport, A, A_new);
        }
        return 1;
    }

    // output: A_new
    NumB math21_geometry_project_perspective_get_camera_matrix_combined_with_viewport(const MatR &A,
                                                                                      const VecR &mat_distortion,
                                                                                      NumN nr_old, NumN nc_old,
                                                                                      MatR &mat_A_new,
                                                                                      NumN nr_new, NumN nc_new,
                                                                                      NumR alpha) {
        NumB flag;
        if (nr_new * nc_new == 0) {
            nr_new = nr_old;
            nc_new = nc_old;
        }

        MatR &A_new = mat_A_new;
        if (!A_new.isSameSize(A.shape())) {
            A_new.setSize(A.shape());
        }

        Interval2D inner;
        Interval2D outer;
        NumB isUseInner = 1, isUseOuter = 1;
        if (alpha == 1) {
            isUseInner = 0;
        } else if (alpha == 0) {
            isUseOuter = 0;
        }

        // Todo: two kinds of methods, but the normalized needs proof.
        NumB isNormalized = 1;
//        NumB isNormalized = 0;
        flag = math21_geometry_project_perspective_get_rectangles_from_corrected(nr_old, nc_old, A, mat_distortion,
                                                                                 MatR(), MatR(), inner, outer,
                                                                                 isUseInner,
                                                                                 isUseOuter, isNormalized);
        if (!flag) {
            return 0;
        }
        // Projection mapping inner rectangle to viewport
        NumR fx0 = 0, fy0 = 0, cx0 = 0, cy0 = 0;
        if (isUseInner) {
            NumR l1, l2;
            if (isNormalized) {
                l1 = inner(1).length();
                l2 = inner(2).length();
            } else {
                l1 = inner(1).lengthDiscrete();
                l2 = inner(2).lengthDiscrete();
            }
            fx0 = nr_new / l1;
            fy0 = nc_new / l2;
            cx0 = -fx0 * inner(1).left();
            cy0 = -fy0 * inner(2).left();
        }

        // Projection mapping outer rectangle to viewport
        NumR fx1 = 0, fy1 = 0, cx1 = 0, cy1 = 0;
        if (isUseOuter) {
            NumR l1, l2;
            if (isNormalized) {
                l1 = outer(1).length();
                l2 = outer(2).length();
            } else {
                l1 = outer(1).lengthDiscrete();
                l2 = outer(2).lengthDiscrete();
            }
            fx1 = nr_new / l1;
            fy1 = nc_new / l2;
            cx1 = -fx1 * outer(1).left();
            cy1 = -fy1 * outer(2).left();
        }

        NumR fx, fy, cx, cy;
        // Interpolate between the two optimal projections
        fx = fx0 * (1 - alpha) + fx1 * alpha;
        fy = fy0 * (1 - alpha) + fy1 * alpha;
        cx = cx0 * (1 - alpha) + cx1 * alpha;
        cy = cy0 * (1 - alpha) + cy1 * alpha;

        MatR viewport(3, 3);
        viewport =
                fx, 0, cx,
                0, fy, cy,
                0, 0, 1;
        if (isNormalized) {
            A_new = viewport;
        } else {
            math21_operator_multiply(1.0, viewport, A, A_new);
        }
        return 1;
    }

    // A is camera matrix of distorted image.
    // distortion is distortion coefficients of distorted image.
    // R is rotation matrix of corrected image.
    // A_new is camera matrix of corrected image.
    // index is index of corrected image.
    // ki are distortion coefficients.
    NumB math21_geometry_project_fisheye_cal_corrected_image_index(const MatR &A,
                                                                   const VecR &mat_distortion,
                                                                   const MatR &mat_R,
                                                                   const MatR &mat_A_new,
                                                                   NumN nr, NumN nc, TenR &index) {
        MATH21_ASSERT(A.isSameSize(3, 3))
        MATH21_ASSERT(mat_A_new.isEmpty() || mat_A_new.isSameSize(3, 3))
        MATH21_ASSERT(mat_R.isEmpty() || mat_R.isSameSize(3, 3))
        MATH21_ASSERT(mat_distortion.isEmpty()
                      || mat_distortion.isSameSize(4) || mat_distortion.isSameSize(4, 1))

        if (!index.isSameSize(2, nr, nc)) {
            index.setSize(2, nr, nc);
        }

        MatR A_new;
        if (mat_A_new.isEmpty()) {
            A_new.setSize(A.shape());
            A_new = A;
        } else {
            A_new.setSize(mat_A_new.shape());
            A_new = mat_A_new;
        }

        MatR R;
        R.setSize(3, 3);
        if (mat_R.isEmpty()) {
            math21_operator_mat_eye(R);
        } else {
            R = mat_R;
        }

        VecR distortion;
        distortion.setSize(4);
        distortion = 0;
        if (!mat_distortion.isEmpty()) {
            math21_operator_container_set_partially(mat_distortion, distortion, 0, 0, mat_distortion.size());
        }

        MatR T_new_inv, T_new;
        math21_operator_multiply(1.0, A_new, R, T_new_inv);
        if (!math21_operator_inverse(T_new_inv, T_new)) {
            return 0;
        }

        NumR u0 = A(1, 3), v0 = A(2, 3);
        NumR fx = A(1, 1), fy = A(2, 2);

        NumR k1, k2, k3, k4;

        k1 = distortion(1);
        k2 = distortion(2);
        k3 = distortion(3);
        k4 = distortion(4);

        VecR p(3), P(3);
        for (NumN i = 1; i <= nr; ++i) {
            for (NumN j = 1; j <= nc; ++j) {
                p = i, j, 1;
                math21_geometry_project(T_new, p, P);
                NumR x = P(1), y = P(2);
                NumR r = xjsqrt(x * x + y * y);
                NumR theta = xjatan(r);

                NumR theta2 = theta * theta, theta4 = theta2 * theta2,
                        theta6 = theta4 * theta2, theta8 = theta4 * theta4;
                NumR theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
                NumR scale = (r == 0) ? 1.0 : theta_d / r;

                NumR xd = x * scale;
                NumR yd = y * scale;
                NumR u = fx * xd + u0;
                NumR v = fy * yd + v0;
                index(1, i, j) = u;
                index(2, i, j) = v;
            }
        }
        return 1;
    }

    void math21_geometry_project_perspective_cal_corrected_image_index_kernel(
            NumN n, NumB noDistortion,
            NumR u0, NumR v0, NumR fx, NumR fy, NumR k1, NumR k2, NumR p1, NumR p2, NumR k3, NumR k4, NumR k5, NumR k6,
            const NumR *T,
            NumR *p,
            NumR *P,
            NumR *index,
            NumN nr,
            NumN nc, NumN id) {
        if (id > n) return;
        NumN i, j;
        math21_device_index_1d_to_2d_fast(&i, &j, id, nc);
        p[1] = i;
        p[2] = j;
        p[3] = 1;
        math21_device_geometry_project_2d(T, p, P);
        NumR x = P[1], y = P[2];
        if (!noDistortion) {
            NumR x2 = x * x, y2 = y * y;
            NumR r2 = x2 + y2, _2xy = 2 * x * y;
            NumR kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2);
            NumR xd = (x * kr + p1 * _2xy + p2 * (r2 + 2 * x2));
            NumR yd = (y * kr + p1 * (r2 + 2 * y2) + p2 * _2xy);
            x = xd;
            y = yd;
        }
        NumR u, v;
        math21_device_geometry_K_project_2d(&u, &v, x, y, fx, fy, u0, v0);
        NumN i_index;
        math21_device_index_3d_to_1d_fast(1, i, j, &i_index, nr, nc);
        index[i_index] = u;
        math21_device_index_3d_to_1d_fast(2, i, j, &i_index, nr, nc);
        index[i_index] = v;
    }

    // see cv::initUndistortRectifyMap
    // x = AX, x' = A'RX, A': 3x3 or 3x4, (choose sub(A') 3x3 if A' 3x4)
    // A is camera K matrix of distorted image.
    // distortion is distortion coefficients of distorted image.
    // R is rotation matrix of corrected image.
    // A_new is camera K matrix or P matrix of corrected image.
    // index is index of corrected image.
    // ki are radial distortion coefficients.
    // pi are tangential distortion coefficients.
    // no thin prism distortion.
    // no trapezoidal distortion.
    // barrel distortion: k1<0.
    // pincushion distortion: k1>0.
    NumB math21_geometry_project_perspective_cal_corrected_image_index(const MatR &A,
                                                                       const VecR &mat_distortion,
                                                                       const MatR &mat_R,
                                                                       const MatR &mat_A_new,
                                                                       NumN nr, NumN nc, TenR &index) {
        MatR R, distortion, A_new;
        math21_geometry_project_matrix_to_standard(A, mat_distortion, mat_R, mat_A_new, distortion, R, A_new);
        if (!index.isSameSize(2, nr, nc)) {
            index.setSize(2, nr, nc);
        }

        MatR T_new_inv, T_new;
        math21_operator_multiply(1.0, A_new, R, T_new_inv);
        if (!math21_operator_inverse(T_new_inv, T_new)) {
            return 0;
        }

        NumR u0 = A(1, 3), v0 = A(2, 3);
        NumR fx = A(1, 1), fy = A(2, 2);

        NumB noDistortion = mat_distortion.isEmpty();
        NumR k1, k2, p1, p2, k3, k4, k5, k6;

        k1 = distortion(1);
        k2 = distortion(2);
        p1 = distortion(3);
        p2 = distortion(4);
        k3 = distortion(5);
        k4 = distortion(6);
        k5 = distortion(7);
        k6 = distortion(8);

        VecR p(3), P(3);
        auto data_T = T_new.getDataAddress();
        auto data_p = p.getDataAddress();
        auto data_P = P.getDataAddress();
        auto data_index = index.getDataAddress();
        data_T -= 1;
        data_p -= 1;
        data_P -= 1;
        data_index -= 1;

        NumN n = nr * nc;
        NumN id;
#pragma omp parallel for
        for (id = 1; id <= n; ++id) {
            math21_geometry_project_perspective_cal_corrected_image_index_kernel(
                    n, noDistortion,
                    u0, v0, fx, fy, k1, k2, p1, p2, k3, k4, k5, k6,
                    data_T,
                    data_p,
                    data_P,
                    data_index,
                    nr,
                    nc, id);
        }
        return 1;
    }

    void math21_geometry_convert_project_between_left_and_right_hand(MatR &T) {
        MATH21_ASSERT(T.isSameSize(3, 3))
        math21_operator_swap_rows(T, 1, 2);
        math21_operator_swap_cols(T, 1, 2);
    }

    // swap p1 and p2
    void math21_geometry_convert_perspective_distortion_between_left_and_right_hand(VecR &distortion) {
        m21_swap(distortion.at(3), distortion.at(4));
    }

    // project points in [x1, x2]*[y1, y2] to [1, nr]*[1, nc].
    // src_index(1), src_index(2) keeps the original x, y position respectively.
    // X = lambda * T * x;
    NumB
    math21_geometry_project_cal_image_index(const MatR &T, NumR x1, NumR x2, NumR y1, NumR y2, NumN nr, NumN nc,
                                            TenR &src_index) {
        NumN i;
        VecR xa(3), xb(3), xc(3), xd(3);
        xa = x1, y1, 1;
        xb = x2, y1, 1;
        xc = x2, y2, 1;
        xd = x1, y2, 1;

        VecR Xa(3), Xb(3), Xc(3), Xd(3);
        math21_geometry_project(T, xa, Xa);
        math21_geometry_project(T, xb, Xb);
        math21_geometry_project(T, xc, Xc);
        math21_geometry_project(T, xd, Xd);

        NumR scale, scale_y, scale_x;
        NumR y_min, y_max, x_min, x_max;
        x_min = Xa(1);
        y_min = Xa(2);
        x_max = x_min;
        y_max = y_min;
        i = 1;
        while (i <= 4) {
            VecR *X_p = 0;
            switch (i) {
                case 1:
                    X_p = &Xa;
                    break;
                case 2:
                    X_p = &Xb;
                    break;
                case 3:
                    X_p = &Xc;
                    break;
                case 4:
                    X_p = &Xd;
                    break;
            }
            VecR &X = *X_p;
            if (x_min > X(1)) {
                x_min = X(1);
            }
            if (y_min > X(2)) {
                y_min = X(2);
            }
            if (x_max < X(1)) {
                x_max = X(1);
            }
            if (y_max < X(2)) {
                y_max = X(2);
            }
            ++i;
        }
//        m21log("warp img border", x_min, y_min, x_max, y_max);
        if (x_min >= x_max || y_min >= y_max) {
            m21error("warp img border error");
            return 0;
        }
        scale_x = nr / (x_max - x_min);
        scale_y = nc / (y_max - y_min);
        scale = xjmin(scale_y, scale_x);

        VecR t(3);
        t = x_min, y_min, 1;
        math21_operator_linear_to(scale, t);

        if (!src_index.isSameSize(2, nr, nc)) {
            src_index.setSize(2, nr, nc);
        }

        MatR T_inv;
        if (!math21_operator_inverse(T, T_inv)) {
            return 0;
        }
        VecR X(3), x(3);
        for (NumN i = 1; i <= nr; ++i) {
            for (NumN j = 1; j <= nc; ++j) {
                X = i, j, 0;
                math21_operator_linear_to_B(1.0, t, 1.0, X);
                math21_operator_linear_to(1 / scale, X);
                math21_geometry_project(T_inv, X, x);
                src_index(1, i, j) = x(1);
                src_index(2, i, j) = x(2);
            }
        }
        return 1;
    }

    namespace detail {
        // planar or interleaved, i.e., nch*nr*nc or nr*nc*nch
        void math21_geometry_warp_image_using_indexes_cpu_kernel(
                NumN n,
                const NumR *a, NumR *b, const NumR *index,
                NumN a_nr, NumN a_nc,
                NumN b_nr, NumN b_nc, NumN b_nch,
                NumR I1_a,
                NumR I1_b,
                NumB I1_include_a,
                NumB I1_include_b,
                NumR I2_a,
                NumR I2_b,
                NumB I2_include_a,
                NumB I2_include_b,
                NumB isInterleaved,
                NumN interpolationFlag,
                NumN id) {
            if (id > n) return;
            NumN i, j;
            math21_device_index_1d_to_2d_fast(&i, &j, id, b_nc);

            NumN ii_index, jj_index;
            math21_device_index_3d_to_1d_fast(1, i, j, &ii_index, b_nr, b_nc);
            math21_device_index_3d_to_1d_fast(2, i, j, &jj_index, b_nr, b_nc);
            if (math21_device_interval2d_is_include(index[ii_index], index[jj_index],
                                                    I1_a, I1_b, I1_include_a, I1_include_b,
                                                    I2_a, I2_b, I2_include_a, I2_include_b)) {
                NumN ii, jj;
                ii = (NumN) index[ii_index]; // negative value excluded.
                jj = (NumN) index[jj_index];
                if (interpolationFlag == m21_flag_interpolation_none) {
                    NumN k;
                    for (k = 1; k <= b_nch; ++k) {
                        NumN index_a, index_b;
                        index_b = math21_device_image_get_1d_index(k, i, j, b_nch, b_nr, b_nc, isInterleaved);
                        index_a = math21_device_image_get_1d_index(k, ii, jj, b_nch, a_nr, a_nc, isInterleaved);
                        b[index_b] = a[index_a];
                    }
                } else if (0) { // remove
                    NumR x, y;
                    x = index[ii_index] - ii;
                    y = index[jj_index] - jj;
                    NumN k;
                    for (k = 1; k <= b_nch; ++k) {
                        NumN index_a00, index_a10, index_a01, index_a11, index_b;
                        NumR f00, f10, f01, f11;
                        index_b = math21_device_image_get_1d_index(k, i, j, b_nch, b_nr, b_nc, isInterleaved);
                        /*
                         * bilinear interpolation
                           Note: Contrary to what the name suggests, the interpolation is not linear.
                           f(x, y) <- f00*(1-x)*(1-y) + f10*x*(1-y) + f01*(1-x)*y + f11*x*y, here f10 :=f(1,0)
                           f(x, y) <- (1-x, x) * A * (1-y,y).t, where A = (f00, f01; f10, f11)
                           f(x, y) <- b1 + b2*x + b3*y + b4*x*y, where b1 = f00, b2 = f10 - f00, b3 = f01-f00, b4 = f00+f11-f10-f01
                         * */
                        index_a00 = math21_device_image_get_1d_index(k, ii, jj, b_nch, a_nr, a_nc, isInterleaved);
                        index_a10 = math21_device_image_get_1d_index(k, ii + 1, jj, b_nch, a_nr, a_nc, isInterleaved);
                        index_a01 = math21_device_image_get_1d_index(k, ii, jj + 1, b_nch, a_nr, a_nc, isInterleaved);
                        index_a11 = math21_device_image_get_1d_index(k, ii + 1, jj + 1, b_nch, a_nr, a_nc,
                                                                     isInterleaved);
                        if (index_a00 == 0 && index_a11 == 0) {
                            continue;
                        }
                        f00 = index_a00 != 0 ? a[index_a00] : a[index_a11];
                        f10 = index_a10 != 0 ? a[index_a10] : f00;
                        f01 = index_a01 != 0 ? a[index_a01] : f00;
                        f11 = index_a11 != 0 ? a[index_a11] : f00;
                        b[index_b] = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + f01 * (1 - x) * y + f11 * x * y;
                    }
                } else if (interpolationFlag == m21_flag_interpolation_bilinear) {
                    NumN k;
                    for (k = 1; k <= b_nch; ++k) {
                        NumN index_b;
                        index_b = math21_device_image_get_1d_index(k, i, j, b_nch, b_nr, b_nc, isInterleaved);
                        NumR value;
                        NumB flag = math21_device_image_get_pixel_bilinear_interpolate(
                                a, &value, k, index[ii_index], index[jj_index],
                                b_nch, a_nr, a_nc, isInterleaved);
                        if (!flag) continue;
                        else b[index_b] = value;
                    }
                }
            }

        }

        void math21_geometry_warp_image_using_indexes_cpu(
                const TenR &A, TenR &B, const TenR &index, const Interval2D &I_, NumB isInterleaved,
                NumN interpolationFlag) {
            MATH21_ASSERT(A.dims() == 3 && index.dims() == 3 && index.dim(1) == 2,
                          "" << A.logInfo("A") << index.logInfo("index"));
            NumN a_nr, a_nc, b_nr, b_nc, a_nch;
            b_nr = index.dim(2);
            b_nc = index.dim(3);
            if (!isInterleaved) {
                a_nr = A.dim(2);
                a_nc = A.dim(3);
                a_nch = A.dim(1);
            } else {
                a_nr = A.dim(1);
                a_nc = A.dim(2);
                a_nch = A.dim(3);
            }
            if (B.isEmpty()) {
                if (!isInterleaved) {
                    B.setSize(a_nch, b_nr, b_nc);
                } else {
                    B.setSize(b_nr, b_nc, a_nch);
                }
            }
            MATH21_ASSERT((!isInterleaved ? (B.isSameSize(a_nch, b_nr, b_nc)) : (B.isSameSize(b_nr, b_nc, a_nch))),
                          "" << A.logInfo("A") << B.logInfo("B"));
            Interval2D I(I_);
            if (I.isEmpty()) {
                I(1).set(0, a_nr, 0, 1);
                I(2).set(0, a_nc, 0, 1);
            }
            auto data_a = A.getDataAddress();
            auto data_b = B.getDataAddress();
            auto data_index = index.getDataAddress();
            data_a -= 1;
            data_b -= 1;
            data_index -= 1;
            NumR I1_a = I(1).left();
            NumR I1_b = I(1).right();
            NumB I1_include_a = I(1).isLeftClosed();
            NumB I1_include_b = I(1).isLeftClosed();
            NumR I2_a = I(2).left();
            NumR I2_b = I(2).right();
            NumB I2_include_a = I(2).isLeftClosed();
            NumB I2_include_b = I(2).isLeftClosed();
            NumN n = b_nr * b_nc;
            NumN id;
#pragma omp parallel for
            for (id = 1; id <= n; ++id) {
                math21_geometry_warp_image_using_indexes_cpu_kernel(
                        n, data_a, data_b, data_index,
                        a_nr, a_nc, b_nr, b_nc, a_nch,
                        I1_a,
                        I1_b,
                        I1_include_a,
                        I1_include_b,
                        I2_a,
                        I2_b,
                        I2_include_a,
                        I2_include_b,
                        isInterleaved, interpolationFlag, id);
            }
        }
    }

    // see cv::remap
    void
    math21_geometry_warp_image_using_indexes_wrapper(const TenR &A, TenR &B, const TenR &index, const Interval2D &I,
                                                     NumB isInterleaved, NumN interpolationFlag) {
#if defined(MATH21_FLAG_USE_CPU)
        detail::math21_geometry_warp_image_using_indexes_cpu(A, B, index, I, isInterleaved, interpolationFlag);
#elif defined(MATH21_FLAG_USE_CUDA)
        MATH21_ASSERT(!I.isEmpty() && isInterleaved==0);
        detail::math21_geometry_warp_image_using_indexes_cuda(A, B, index, I);
#elif defined(MATH21_FLAG_USE_OPENCL)
        MATH21_ASSERT(0)
#endif
    }

    // see cv::remap
    void math21_geometry_warp_image_using_indexes(const TenR &A, TenR &B, const TenR &index, const Interval2D &I,
                                                  NumB isInterleaved, NumN interpolationFlag) {
        if (A.is_cpu()) {
            detail::math21_geometry_warp_image_using_indexes_cpu(A, B, index, I, isInterleaved, interpolationFlag);
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            math21_geometry_warp_image_using_indexes_wrapper(A, B, index, I, isInterleaved, interpolationFlag);
#else
            MATH21_ASSERT(0);
#endif
        }
    }

    void math21_geometry_warp_image_using_project_mat(const MatR &src, MatR &dst,
                                                      const MatR &A, const MatR &distortion, const MatR &R,
                                                      const MatR &P,
                                                      NumN nr_new, NumN nc_new, NumB isInterleaved,
                                                      NumN interpolationFlag) {
        MatR index;
        math21_geometry_project_perspective_cal_corrected_image_index(A, distortion, R, P,
                                                                      nr_new, nc_new, index);
        math21_geometry_warp_image_using_indexes(src, dst, index, Interval2D(), isInterleaved, interpolationFlag);
    }

    // User should make sure points in A lie in index region.
    void math21_geometry_project_points_using_indexes(const MatR &A, MatR &B, const TenR &index) {
        MATH21_ASSERT(A.ncols() == 2)
        if (!B.isSameSize(A.nrows(), 2)) {
            B.setSize(A.nrows(), 2);
        }
        for (NumN k = 1; k <= A.nrows(); ++k) {
            MATH21_ASSERT(A(k, 1) >= 0)
            MATH21_ASSERT(A(k, 2) >= 0)
            NumN i = (NumN) A(k, 1);
            NumN j = (NumN) A(k, 2);
            B(k, 1) = index(1, i, j);
            B(k, 2) = index(2, i, j);
        }
    }

    //scale x in [a,b] to [c, d]
    void scale(const MatR &m, MatR &C, NumR a, NumR b, NumR c, NumR d) {
        NumN i, j;
        NumN nr = m.nrows();
        NumN nc = m.ncols();
        MATH21_ASSERT(b > a && d > c, "scale range negative");
        MATH21_ASSERT(nr >= 1 && nc >= 1, "empty matrix");
        if (!C.isSameSize(m.shape())) {
            C.setSize(nr, nc);
        }
        NumR s = (d - c) / (b - a);
        for (i = 1; i <= nr; i++) {
            for (j = 1; j <= nc; j++) {
                C(i, j) = (m(i, j) - a) * s + c;
            }
        }
    }

    //scale x in [a,b] to [c, d], Y can't be X. if element of matrix X is all same, and c=0, d=1, then X will be 0.5.
    void scale(const TenR &X, TenR &Y, NumR c, NumR d) {
        MATH21_ASSERT(X.dims() == 2 && X.dim(2) == 1, "not satisfied in current version!")
        MATH21_ASSERT(d > c, "scale range negative");
        NumN i1, i2;
        NumR a, b;
        a = X(1, 1);
        b = X(1, 1);
        NumR tmp;
        for (i1 = 1; i1 <= X.dim(1); i1++) {
            for (i2 = 1; i2 <= X.dim(2); i2++) {
                tmp = X(i1, i2);
                if (tmp < a)a = tmp;
                if (tmp > b)b = tmp;
            }
        }
        if (!Y.isSameSize(X.shape())) {
            Y.setSize(X.shape());
        }
        if (b == a) {
            tmp = (d + c) / 2;
            for (i1 = 1; i1 <= X.dim(1); i1++) {
                for (i2 = 1; i2 <= X.dim(2); i2++) {
                    Y(i1, i2) = tmp;
                }
            }
        } else {
            NumR s = (d - c) / (b - a);
            for (i1 = 1; i1 <= X.dim(1); i1++) {
                for (i2 = 1; i2 <= X.dim(2); i2++) {
                    Y(i1, i2) = (X(i1, i2) - a) * s + c;
                }
            }
        }
    }

    void sample_sine_voice(TenR &X, TenR &Y, NumN seconds) {
        const NumR pi2 = XJ_PI * 2;
        const NumN sample_freq = 44100;
//        const NumN sample_freq = 10;
        const NumN n_samples = seconds * sample_freq;
//        uint16_t  ampl;
//        uint8_t bytes[2];
        X.setSize(n_samples, 1);
        Y.setSize(n_samples, 1);
        for (NumN i = 1, t = 0; i <= n_samples; i++, t++) {
//            ampl =(uint16_t)( NumN16_MAX * 0.5 * (1.0 + sin(pi2 * t * 1000.0 / sample_freq)));
//            bytes[0] = ampl>>8;
//            bytes[1] = ampl&0xff;
            X(i, 1) = t;
//            Y(i, 1) = NumN16_MAX * 0.5 * (1.0 + sin(pi2 * t * 1000.0 / sample_freq));
            Y(i, 1) = NumN16_MAX * 0.5 * (1.0 + sin(pi2 * t / sample_freq));
        }
    }

    void math21_sample_heart(MatR &A) {
        NumN n = 5000;
        A.setSize(n, 2);
        NumR t;
        for (NumN i = 1; i <= n; ++i) {
            t = 2.0 * MATH21_PI * i / n;
            A(i, 1) = 16 * m21_pow(xjsin(t), 3);
            A(i, 2) = 13 * xjcos(t) - 5 * xjcos(2 * t) - 2 * xjcos(3 * t) - xjcos(4 * t);
        }
    }

    //draw data to matrix with right hand axis.
    //matrix_axis is a matrix with elements 0 or 1, axis_x, axis_y is size of the matrix.
    void
    draw_data_at_matrix_axis(const TenR &X_sample, const TenR &Y_sample, TenN &matrix_axis, NumN axis_x,
                             NumN axis_y) {
        if (X_sample.isEmpty()) {
            return;
        }
        MATH21_ASSERT(X_sample.dims() == 2 && X_sample.dim(2) == 1 &&
                      math21_operator_container_isEqual(X_sample.shape(), Y_sample.shape()),
                      "not satisfied in current version!")
        TenR X, Y;
        matrix_axis.setSize(axis_x, axis_y);
        scale(X_sample, X, 1, axis_x);
        scale(Y_sample, Y, 1, axis_y);
        NumN j1, j2;
        for (NumN i1 = 1; i1 <= X.dim(1); i1++) {
            for (NumN i2 = 1; i2 <= X.dim(2); i2++) {
                j1 = (NumN) X(i1, i2);
                j2 = (NumN) Y(i1, i2);
                if (matrix_axis(j1, j2) == 0) {
                    matrix_axis(j1, j2) = 1;
                }
            }
        }
    }

    /////////////////////////

    //! returns 3x3 affine transformation matrix for the planar scale and translation.
    // [a1, b1]*[a2,b2] -> [c1, d1]*[c2, d2]
    void la_getScaleTranslationMatrix2D(MatR &M,
                                        NumR a1, NumR b1, NumR a2, NumR b2,
                                        NumR c1, NumR d1, NumR c2, NumR d2) {
        if (c1 == d1) {
            d1 = c1 + 1;
        }
        if (c2 == d2) {
            d2 = c2 + 1;
        }

        MATH21_ASSERT(a1 <= b1 && a2 <= b2 && c1 < d1 && c2 < d2);
        NumR k1, t1, k2, t2;
        if (a1 == b1) {
            k1 = 1;
        } else {
            k1 = (d1 - c1) / (b1 - a1);
        }
        if (a2 == b2) {
            k2 = 1;
        } else {
            k2 = (d2 - c2) / (b2 - a2);
        }
        t1 = c1 - k1 * a1;
        t2 = c2 - k2 * a2;
        math21_la_2d_matrix_scale_and_translate(k1, k2, t1, t2, M);
    }

    void math21_la_getScaleTranslationMatrix2D(MatR &M,
                                               const Interval2D &input,
                                               Interval2D &output) {
        la_getScaleTranslationMatrix2D(M,
                                       input(1).left(), input(1).right(), input(2).left(), input(2).right(),
                                       output(1).left(), output(1).right(), output(2).left(), output(2).right());
    }

    void math21_la_2d_matrix_compute_matrix_axis_to_matrix(NumN axis_x, NumN axis_y, MatR &T_final) {
        MatR T;
        if (!T_final.isSameSize(3, 3)) {
            T_final.setSize(3, 3);
        }
        math21_operator_mat_eye(T_final);

        math21_la_2d_matrix_rotate(MATH21_PI_2, T);
        math21_operator_multiply_to_B(1, T, T_final);
        math21_la_2d_matrix_translate(axis_y, 0, T);
        math21_operator_multiply_to_B(1, T, T_final);
    }

    void math21_la_2d_matrix_compute_matrix_to_matrix_axis(NumN axis_x, NumN axis_y, MatR &T_final) {
        MatR T;
        if (!T_final.isSameSize(3, 3)) {
            T_final.setSize(3, 3);
        }
        math21_operator_mat_eye(T_final);

        math21_la_2d_matrix_translate(-(NumZ) axis_y, 0, T);
        math21_operator_multiply_to_B(1, T, T_final);
        math21_la_2d_matrix_rotate(-MATH21_PI_2, T);
        math21_operator_multiply_to_B(1, T, T_final);
    }

    void la_data_2d_bound(const MatR &A,
                          NumR &a1, NumR &b1, NumR &a2, NumR &b2) {
        MATH21_ASSERT(A.dims() == 2 && A.dim(2) == 2)
        NumN i;
        NumN nr = A.dim(1);
        a1 = A(1, 1);
        a2 = A(1, 2);
        b1 = a1;
        b2 = a2;
        NumR tmp;
        for (i = 1; i <= nr; ++i) {
            tmp = A(i, 1);
            if (tmp < a1) {
                a1 = tmp;
            } else if (tmp > b1) {
                b1 = tmp;
            }
            tmp = A(i, 2);
            if (tmp < a2) {
                a2 = tmp;
            } else if (tmp > b2) {
                b2 = tmp;
            }
        }
    }

    // A: nr * (x, y) or y with x setting to index.
    void math21_la_data_2d_bound(const TenR &A,
                                 NumR &a1, NumR &b1, NumR &a2, NumR &b2) {
        MATH21_ASSERT((A.dims() == 2 && A.dim(2) == 2)
                      || (A.dims() == 1))

        if (A.dims() == 2) {
            TenR B;
            VecN index(A.dims());
            index = 1, 0;

            TensorFunction_min f_min;
            math21_operator_tensor_f_shrink(A, B, index, f_min);
            a1 = B(1);
            a2 = B(2);

            TensorFunction_max f_max;
            math21_operator_tensor_f_shrink(A, B, index, f_max);
            b1 = B(1);
            b2 = B(2);
        } else {
            TenR B;
            VecN index(A.dims());
            index = 1;

            TensorFunction_min f_min;
            math21_operator_tensor_f_shrink(A, B, index, f_min);
            a1 = 1;
            a2 = B(1);

            TensorFunction_max f_max;
            math21_operator_tensor_f_shrink(A, B, index, f_max);
            b1 = A.dim(1);
            b2 = B(1);
        }

    }

    // A: nr * (x, y) or y with x setting to index.
    void math21_la_data_2d_bound(const TenR &A,
                                 Interval2D &I) {
        NumR a1, b1, a2, b2;
        math21_la_data_2d_bound(A, a1, b1, a2, b2);
        I(1).set(a1, b1, 1, 1);
        I(2).set(a2, b2, 1, 1);
    }

    // A: batch * nr * (x, y) or batch * y with x setting to index.
    void math21_la_data_2d_bound_in_batch(const TenR &A,
                                          NumR &a1, NumR &b1, NumR &a2, NumR &b2) {
        MATH21_ASSERT((A.dims() == 3 && A.dim(3) == 2)
                      || (A.dims() == 2))

        if (A.dims() == 3) {
            TenR B;
            VecN index(A.dims());
            index = 1, 1, 0;

            TensorFunction_min f_min;
            math21_operator_tensor_f_shrink(A, B, index, f_min);
            a1 = B(1);
            a2 = B(2);

            TensorFunction_max f_max;
            math21_operator_tensor_f_shrink(A, B, index, f_max);
            b1 = B(1);
            b2 = B(2);
        } else {
            TenR B;
            VecN index(A.dims());
            index = 1, 1;

            TensorFunction_min f_min;
            math21_operator_tensor_f_shrink(A, B, index, f_min);
            a1 = 1;
            a2 = B(1);

            TensorFunction_max f_max;
            math21_operator_tensor_f_shrink(A, B, index, f_max);
            b1 = A.dim(2);
            b2 = B(1);
        }

    }

    // A: batch * nr * (x, y) or batch * y with x setting to index.
    void math21_la_data_2d_bound_in_batch(const TenR &A,
                                          Interval2D &I) {
        NumR a1, b1, a2, b2;
        math21_la_data_2d_bound_in_batch(A, a1, b1, a2, b2);
        I(1).set(a1, b1, 1, 1);
        I(2).set(a2, b2, 1, 1);
    }

    void la_scale_data_2d(const MatR &data, MatR &data_new,
                          NumR c1, NumR d1, NumR c2, NumR d2) {
        if (data.isEmpty()) {
            return;
        }
        MATH21_ASSERT(data.dims() == 2 && data.dim(2) == 2)
        MATH21_ASSERT(c1 < d1 && c2 < d2);
        NumR a1, b1, a2, b2;
        la_data_2d_bound(data, a1, b1, a2, b2);
        MatR M;
        la_getScaleTranslationMatrix2D(M, a1, b1, a2, b2,
                                       c1, d1, c2, d2);
        if (!data_new.isSameSize(data.shape())) {
            data_new.setSize(data.shape());
        }
        NumN nr = data.dim(1);
        NumN i;
        VecR x(3), y(3);
        x(3) = 1;
        for (i = 1; i <= nr; ++i) {
            x(1) = data(i, 1);
            x(2) = data(i, 2);
            math21_operator_multiply(1, M, x, y);
            data_new(i, 1) = y(1);
            data_new(i, 2) = y(2);
        }
    }

    // data: (x, y) or y with x setting to index.
    void la_scale_data_2d(const MatR &data, MatR &data_new,
                          const MatR &M) {
        if (data.isEmpty()) {
            return;
        }
        MATH21_ASSERT((data.dims() == 2 && data.dim(2) == 2) || data.dims() == 1)
        MATH21_ASSERT(M.isSameSize(3, 3));
        NumN nr = data.dim(1);
        if (!data_new.isSameSize(nr, 2)) {
            data_new.setSize(nr, 2);
        }
        NumN i;
        VecR x(3), y(3);
        x(3) = 1;
        if (data.dims() == 2) {
            for (i = 1; i <= nr; ++i) {
                x(1) = data(i, 1);
                x(2) = data(i, 2);
                math21_operator_multiply(1, M, x, y);
                data_new(i, 1) = y(1);
                data_new(i, 2) = y(2);
            }
        } else {
            for (i = 1; i <= nr; ++i) {
                x(1) = i;
                x(2) = data(i);
                math21_operator_multiply(1, M, x, y);
                data_new(i, 1) = y(1);
                data_new(i, 2) = y(2);
            }
        }
    }

    void la_scale_data_2d(MatR &data,
                          const MatR &M) {
        MatR data_new;
        la_scale_data_2d(data, data_new, M);
        data.swap(data_new);
    }

    void la_draw_data_2d_at_matrix_axis_no_scale(const MatR &data, TenN &matrix_axis) {
        if (data.isEmpty()) {
            return;
        }
        MATH21_ASSERT(data.dims() == 2 && data.dim(2) == 2,
                      "not satisfied in current version!")
        MATH21_ASSERT(matrix_axis.dims() == 2 && matrix_axis.dim(1) > 0 && matrix_axis.dim(2) > 0)
        NumN axis_x = matrix_axis.dim(1);
        NumN axis_y = matrix_axis.dim(2);

        NumN j1, j2;
        NumN nr;
        nr = data.dim(1);

//#pragma omp parallel for private(j1, j2)
        for (NumN i1 = 1; i1 <= nr; i1++) {
            j1 = (NumN) data(i1, 1);
            j2 = (NumN) data(i1, 2);
            if (xjIsIn(j1, 1, axis_x) && xjIsIn(j2, 1, axis_y)) {
                if (matrix_axis(j1, j2) == 0) {
                    matrix_axis(j1, j2) = 1;
                }
            }
        }
    }

    // from Wm. Randolph Franklin (https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html)
    NumB math21_geometry_polygon_include_points(const math21::MatR &polygon, NumR x, NumR y) {
        if (polygon.size() < 3)return 0;
        NumB isInside = 0;
        NumR x1, x2, y1, y2;
        for (NumN i2 = 1, i1 = polygon.nrows(); i2 <= polygon.nrows(); ++i2) {
            x1 = polygon(i1, 1);
            y1 = polygon(i1, 2);
            x2 = polygon(i2, 1);
            y2 = polygon(i2, 2);
            if ((y2 > y) != (y1 > y)) {
                if (x < x2 + (x1 - x2) * (y - y2) / (y1 - y2)) {
                    isInside = !isInside;
                }
            }
            i1 = i2;
        }
        return isInside;
    }

    // see Chapter 12, HZ, R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, Cambridge Univ. Press, 2003.
    // AX = 0, where A = (x * P3.t - P1.t; y * P3.t - P2.t; x' * P'3.t - P'1.t; y' * P'3.t - P'2.t)
    // P.t = (P1, P2, P3), P'.t = (P'1, P'2, P'3)
    void math21_geometry_linear_triangulation(const MatR &x1s, const MatR &x2s, MatR &Xs,
                                              const MatR &P1, const MatR &P2) {
        NumN n = x1s.nrows();
        MATH21_ASSERT(x2s.nrows() == n);
        MATH21_ASSERT(P1.isSameSize(3, 4) && P2.isSameSize(3, 4));

        Seqce<const MatR *> xs(2);
        xs = &x1s, &x2s;
        Seqce<const MatR *> Ps(2);
        Ps = &P1, &P2;

        Xs.setSize(n, 4);
        MatR A(4, 4), u, w, v;
        VecR X;
        for (NumN i = 1; i <= n; ++i) {// for each point
            for (NumN j = 1; j <= 2; ++j) {// for each view
                const MatR &x = *xs(j);
                const MatR &P = *Ps(j);
                NumR num_x = x(i, 1);
                NumR num_y = x(i, 2);
                NumN offset = (j - 1) * 2;
                for (NumN k = 1; k <= 4; ++k) {
                    A(offset + 1, k) = num_x * P(3, k) - P(1, k);
                    A(offset + 2, k) = num_y * P(3, k) - P(2, k);
                }
            }
            math21_operator_svd_real(A, u, w, v);
            math21_op_matrix_get_col(X, v, 4);
            math21_op_matrix_set_row(X, Xs, i);
        }
    }
}