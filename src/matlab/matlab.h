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

    namespace matlab {

        // y = k * x
        TenR operator*(NumR k, const TenR &x);

        TenR operator+(const NumR &x1, const TenR &x2);

        TenR operator+(const TenR &x1, const NumR &x2);

        TenR operator+(const TenR &x1, const TenR &x2);

        TenR operator-(const TenR &x1, const TenR &x2);

        TenR operator-(const TenR &x1);

        TenR operator*(const TenR &x1, const TenR &x2);

        TenR multiply_elewise(const TenR &x1, const TenR &x2);

        TenR transpose(const TenR &x1);

        TenR inv(const TenR &x1);

        NumR dot(const TenR &x1, const TenR &x2);

        TenR sin(const TenR &x);

        TenR cos(const TenR &x);

        TenR asin(const TenR &x);

        TenR acos(const TenR &x);

        TenR mod(const TenR &x, NumR m);

        TenR mod(const TenR &x, const TenR &m);

        TenR abs(const TenR &x);

        TenR rand(NumN d1, NumN d2);

        TenR zeros(NumN d1, NumN d2);

        TenR eye(NumN d1);

        TenR eye(NumN d1, NumN d2);

        NumR norm(const MatR &X, NumN p = 2);

        NumR norm(const MatR &X, const std::string &type);

        MatR operator^(const MatR &A, NumZ n);

        MatR diag(const MatR &A, NumZ n = 0);

        NumR trace(const MatR &A);
    }

}