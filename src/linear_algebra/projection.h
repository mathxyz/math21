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

#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
namespace math21 {

    struct View3d {
    private:
        VecR eye;
        VecR u;
        VecR v;
        VecR n;
        NumR aspectRatio;
        NumR viewAngle;
        NumR near;
        NumR far;

        MatR modelView, projection, viewport;
        MatR T;

        void init();

        void setT();

        void setModelViewMatrix();

        void rotateAxes(VecR &u, VecR &v, NumR theta) const;

        // 0 < N < F
        // here aspectRatio = width/height
        void setPerspectiveMatrix(NumR aspectRatio, NumR viewAngle, NumR N, NumR F);

        // see glFrustum( GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val );
        void setFrustum(NumR l, NumR r, NumR b, NumR t, NumR N, NumR F);

    public:

        View3d();

        View3d(const View3d &view);

        View3d &operator=(const View3d &view);

        virtual ~View3d();

        void reset();

        // see glTranslatef
        void translatePoint(const VecR &t);

        // see glRotatef( GLfloat angle, GLfloat x, GLfloat y, GLfloat z );
        // It can be used to rotate 3D models by an angle (radian) about a rotation axis (x, y, z)
        // It is only used for modelView matrix.
        void rotatePoint(const VecR &axis, NumR angle);

        void rotateAboutX(NumR angle);

        void rotateAboutY(NumR angle);

        void rotateAboutZ(NumR angle);

        // equivalent to gluLookAt
        void set(const VecR &eye, const VecR &look, const VecR &up);

        // see gluPerspective (GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar);
        // here aspectRatio = width/height
        // viewAngle is in radian
        void setShape(NumR aspectRatio, NumR viewAngle, NumR near, NumR far);

        void slide(const VecR &delta);

        void roll(NumR theta);

        void pitch(NumR theta);

        void yaw(NumR theta);

        // see glViewport
        // [a1, b1] * [a2, b2]
        void setViewportMatrix(NumR a1, NumR b1, NumR a2, NumR b2);

        NumB project(const VecR &x, VecR &X) const;

        const MatR &getModelView() const;

        const MatR &getProjection() const;

        const MatR &getViewport() const;

        const MatR &getT() const;

        void log(const char *s = 0, NumB noEndl = 0) const;

        void log(std::ostream &io, const char *s = 0, NumB noEndl = 0) const;
    };

    void math21_la_3d_matrix_test(const TenR &A, TenR &B);
}