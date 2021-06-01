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

#include "../../includes/math21_feature_config.h"

// this file can be included in opencl source file
// Support for double precision floating-point type double in opencl kernels requires an extension.

#ifdef __cplusplus
extern "C" {
#endif

// Type used:
typedef int NumZ;
typedef unsigned int NumN;

#if defined(MATH21_FLAG_IS_WIN32) && defined(MATH21_FLAG_USE_OPENCL)
#define MATH21_USE_NUMR32
typedef float NumR; // default floating type when using opencl, or in mobile devices.
#else
typedef double NumR; // recommended default floating type, must enable this when called by python.
#endif

// Type not used
typedef NumN NumB;

// put here for opencl only.
typedef float NumR32;

#ifdef __cplusplus
}
#endif
