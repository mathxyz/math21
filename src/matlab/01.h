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

    void math21_operator_container_create_with_increment(VecR &v, NumR from, NumR inc, NumR to, NumR epsilon, NumN n);

    void math21_matlab_vector_create_with_increment(VecR &v, NumR from, NumR inc, NumR to, NumR epsilon = 0);

    void math21_matlab_vector_create_with_unit_increment(VecR &v, NumR from, NumR to);

    void math21_matlab_mpower(const MatR &A, NumZ n, MatR &y);

    void math21_matlab_diag(const MatR &A, MatR &B, NumZ n = 0);

    void math21_la_convert_RodriguesForm_to_rotation(const VecR &w, MatR &R);

    void math21_la_convert_rotation_to_RodriguesForm(const MatR &R, VecR &w);

}