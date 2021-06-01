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

#include "inner_src.h"
#include "AffineTransform.h"
#include "projection.h"

namespace math21 {
    void View3d::init() {
        eye.setSize(3);
        u.setSize(3);
        v.setSize(3);
        n.setSize(3);
        aspectRatio = 0;
        viewAngle = 0;
        near = 0;
        far = 0;
        modelView.setSize(4, 4);
        projection.setSize(4, 4);
        viewport.setSize(4, 4);
        T.setSize(4, 4);

        math21_operator_mat_eye(modelView);
        math21_operator_mat_eye(projection);
        math21_operator_mat_eye(viewport);
        math21_operator_mat_eye(T);
    }

    void View3d::reset() {
        eye = 0;
        u = 0;
        v = 0;
        n = 0;
        aspectRatio = 0;
        viewAngle = 0;
        near = 0;
        far = 0;
        math21_operator_mat_eye(modelView);
        math21_operator_mat_eye(projection);
        math21_operator_mat_eye(viewport);
        math21_operator_mat_eye(T);
    }

    void View3d::setT() {
        math21_operator_mat_eye(T);
        math21_operator_multiply_to_B(1, modelView, T);
        math21_operator_multiply_to_B(1, projection, T);
        math21_operator_multiply_to_B(1, viewport, T);
    }

    void View3d::setModelViewMatrix() {
        NumR t1 = -math21_operator_container_InnerProduct(1, eye, u);
        NumR t2 = -math21_operator_container_InnerProduct(1, eye, v);
        NumR t3 = -math21_operator_container_InnerProduct(1, eye, n);
        MatR MV(4, 4);
        MV =
                u(1), u(2), u(3), t1,
                v(1), v(2), v(3), t2,
                n(1), n(2), n(3), t3,
                0, 0, 0, 1;
        math21_operator_multiply_to_B(1, MV, modelView);
        setT();
    }

    void View3d::rotateAxes(VecR &u, VecR &v, NumR theta) const {
        VecR u_prime(3);
        VecR v_prime(3);
        math21_operator_container_linear(xjcos(theta), u, xjsin(theta), v, u_prime);
        math21_operator_container_linear(-xjsin(theta), u, xjcos(theta), v, v_prime);
        u = u_prime;
        v = v_prime;
    }

    View3d::View3d() {
        init();
    }

    View3d::View3d(const View3d &view) {
        init();
        *this = view;
    }

    View3d &View3d::operator=(const View3d &view) {
        eye = view.eye;
        u = view.u;
        v = view.v;
        n = view.n;
        aspectRatio = view.aspectRatio;
        viewAngle = view.viewAngle;
        near = view.near;
        far = view.far;
        modelView = view.modelView;
        projection = view.projection;
        viewport = view.viewport;
        T = view.T;
        return *this;
    }

    View3d::~View3d() {
    }

    // modelView <- MT * modelView
    void View3d::translatePoint(const VecR &t) {
        MatR MT(4, 4);
        MT =
                1, 0, 0, t(1),
                0, 1, 0, t(2),
                0, 0, 1, t(3),
                0, 0, 0, 1;
        math21_operator_multiply_to_B(1, MT, modelView);
        setT();
    }

    // Todo: need proof.
    // modelView <- R * modelView
    void View3d::rotatePoint(const VecR &axis, NumR angle) {
        NumR x, y, z, c, s;
        x = axis(1);
        y = axis(2);
        z = axis(3);
        c = xjcos(angle);
        s = xjsin(angle);
        MatR R(4, 4);
        R =
                x * x * (1 - c) + c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0,
                x * y * (1 - c) + z * s, y * y * (1 - c) + c, y * z * (1 - c) - x * s, 0,
                x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z * z * (1 - c) + c, 0,
                0, 0, 0, 1;
        math21_operator_multiply_to_B(1, R, modelView);
        setT();
    }

    void View3d::rotateAboutX(NumR angle) {
        VecR n(3);
        n = 1, 0, 0;
        rotatePoint(n, angle);
    }

    void View3d::rotateAboutY(NumR angle) {
        VecR n(3);
        n = 0, 1, 0;
        rotatePoint(n, angle);
    }

    void View3d::rotateAboutZ(NumR angle) {
        VecR n(3);
        n = 0, 0, 1;
        rotatePoint(n, angle);
    }

    void View3d::set(const VecR &eye, const VecR &look, const VecR &up) {
        this->eye = eye;
        math21_operator_container_subtract_to_C(eye, look, n);
        math21_operator_container_CrossProduct(up, n, u);
        math21_operator_container_normalize_to_A(n, 2);
        math21_operator_container_normalize_to_A(u, 2);
        math21_operator_container_CrossProduct(n, u, v);
        setModelViewMatrix();
    }

    void View3d::setShape(NumR aspectRatio, NumR viewAngle, NumR near, NumR far) {
        this->aspectRatio = aspectRatio;
        this->viewAngle = viewAngle;
        this->near = near;
        this->far = far;
        setPerspectiveMatrix(aspectRatio, viewAngle, near, far);
    }

    void View3d::setFrustum(NumR l, NumR r, NumR b, NumR t, NumR N, NumR F) {
        projection =
                2 * N / (r - l), 0, (r + l) / (r - l), 0,
                0, 2 * N / (t - b), (t + b) / (t - b), 0,
                0, 0, (N + F) / (N - F), (2 * N * F) / (N - F),
                0, 0, -1, 0;
        setT();
    }

    // 0 < N < F
    // Note: here we use aspectRatio = nr/nc?
    // here aspectRatio = width/height = x/y in view plane.
    // the aspect ratio here should match the aspect ratio of the associated viewport in order to avoid distortion.
    void View3d::setPerspectiveMatrix(NumR aspectRatio, NumR viewAngle, NumR N, NumR F) {
        NumR l;
        NumR r;
        NumR b;
        NumR t;
        t = N * xjtan(viewAngle / 2);
        b = -t;
        r = t * aspectRatio;
        l = -r;
        setFrustum(l, r, b, t, N, F);
    }

    void View3d::slide(const VecR &delta) {
        math21_operator_container_linear_to_A(1, eye, delta(1), u, delta(2), v, delta(3), n);
        setModelViewMatrix();
    }

    void View3d::roll(NumR theta) {
        rotateAxes(u, v, theta);
        setModelViewMatrix();
    }

    void View3d::pitch(NumR theta) {
        rotateAxes(v, n, theta);
        setModelViewMatrix();
    }

    void View3d::yaw(NumR theta) {
        rotateAxes(n, u, theta);
        setModelViewMatrix();
    }

    void View3d::setViewportMatrix(NumR a1, NumR b1, NumR a2, NumR b2) {
        MATH21_ASSERT(b1 > a1 && b2 > a2)
        NumR s1, s2, t1, t2;
        s1 = (b1 - a1) / 2;
        s2 = (b2 - a2) / 2;
        t1 = (a1 + b1) / 2;
        t2 = (a2 + b2) / 2;
        viewport =
                s1, 0, 0, t1,
                0, s2, 0, t2,
                0, 0, 0.5, 0.5,
                0, 0, 0, 1;
        setT();
    }

    NumB View3d::project(const VecR &x, VecR &X) const {
        return math21_geometry_project_3d(T, x, X);
    }

    // equivalent to glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    const MatR &View3d::getModelView() const {
        return modelView;
    }

    const MatR &View3d::getProjection() const {
        return projection;
    }

    const MatR &View3d::getViewport() const {
        return viewport;
    }

    const MatR &View3d::getT() const {
        return T;
    }

    void View3d::log(const char *s, NumB noEndl) const {
        log(std::cout, s, noEndl);
    }

    void View3d::log(std::ostream &io, const char *s, NumB noEndl) const {
        if (s) {
            io << "View3d " << s << ":\n";
        }
        eye.log("eye");
        u.log("u");
        v.log("v");
        n.log("n");
        io
                << "aspectRatio: " << aspectRatio << "\n"
                << "viewAngle : " << viewAngle << "\n"
                << "near: " << near << "\n"
                << "far: " << far << "\n";
        modelView.log("modelView");
        projection.log("projection");
        viewport.log("viewport");
        T.log("T");
        if (!noEndl) {
            io << std::endl;
        }
    }

    void math21_la_3d_matrix_test_0(const TenR &A, TenR &B) {
        // get matrix
        MatR T_final, T;
        T_final.setSize(4, 4);
        math21_operator_mat_eye(T_final);

        math21_la_3d_matrix_translate(50, 0, 0, T);
        math21_operator_multiply_to_B(1, T, T_final);

        NumR angle;
        angle = MATH21_PI / 10;
//        math21_la_3d_matrix_rotate_about_x_axis(angle, T);
        math21_la_3d_matrix_rotate_about_y_axis(angle, T);
//        math21_la_3d_matrix_rotate_about_z_axis(angle, T);
        math21_operator_multiply_to_B(1, T, T_final);

        // view3d
        View3d view3d;
        VecR eye(3);
        VecR look(3);
        VecR up(3);
        eye = 20, 20, 180;
        look = 0, 0, 0;
        up = 0, 1, 0;
        view3d.set(eye, look, up);
        view3d.setShape(1, MATH21_PI / 2, 2, 200);
        view3d.setViewportMatrix(1, 100, 1, 100);
        math21_operator_multiply_to_B(1, view3d.getT(), T_final);

        // transform
        if (!math21_la_3d_perspective_projection_image(A, B, T_final)) {
            return;
        }
    }

    void math21_la_3d_matrix_test_2(const TenR &A, TenR &B) {
        // get matrix
        MatR T_final, T;
        T_final.setSize(4, 4);
        math21_operator_mat_eye(T_final);

        math21_la_3d_matrix_translate(50, 0, 0, T);
        math21_operator_multiply_to_B(1, T, T_final);

        NumR angle;
        angle = MATH21_PI / 10;
//        math21_la_3d_matrix_rotate_about_x_axis(angle, T);
        math21_la_3d_matrix_rotate_about_y_axis(angle, T);
//        math21_la_3d_matrix_rotate_about_z_axis(angle, T);
        math21_operator_multiply_to_B(1, T, T_final);

        //
        NumR width = 600;
        NumR height = 600;
        angle = 75;

        // view3d
        View3d view3d;
        VecR t(3);
        t = 0, 0, 0;
        view3d.translatePoint(t);
        view3d.getModelView().log("modelView");


        view3d.rotateAboutZ(xjdegree2radian(23.0));

        view3d.rotateAboutY(xjdegree2radian(0));
        view3d.rotateAboutX(xjdegree2radian(0));

        view3d.getModelView().log("modelView");

        VecR eye(3);
        VecR look(3);
        VecR up(3);
        eye = 0, 0, 15;
        look = 0, 0, 0;
        up = 0, 1, 0;
        view3d.set(eye, look, up);
        view3d.getModelView().log("modelView");

        view3d.setShape(width / height, xjdegree2radian(angle), 0.1, 100);
        view3d.getProjection().log("Projection", 0, 1, 6);

        view3d.setViewportMatrix(1, 100, 1, 100);
        math21_operator_multiply_to_B(1, view3d.getT(), T_final);

        // transform
//        if (!math21_la_3d_perspective_projection_image(A, B, T_final)) {
//            return;
//        }
    }

    void math21_la_3d_matrix_test(const TenR &A, TenR &B) {
    }
}