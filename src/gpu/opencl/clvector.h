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
#include "opencl_c.h"
#include "inner_c.h"

#ifdef MATH21_FLAG_USE_OPENCL

// offset can be negative.
m21clvector math21_opencl_vector_pointer_increase(m21clvector buffer, int offset);

m21clvector operator+(m21clvector, size_t);

m21clvector operator-(m21clvector, size_t);

m21clvector &operator+=(m21clvector &, size_t);

m21clvector &operator-=(m21clvector &, size_t);

bool operator==(const m21clvector &x, const m21clvector &y);

bool operator!=(const m21clvector &x, const m21clvector &y);

NumB math21_opencl_vector_isEmpty(const m21clvector *clvector);

void math21_opencl_vector_reset(m21clvector *clvector);

// offset can be negative.
void math21_opencl_vector_set(m21clvector *clvector, size_t size, cl_mem buffer, size_t size_address, cl_mem address,
                              int offset, int unit);

void math21_opencl_vector_free(m21clvector x_gpu);

#endif
