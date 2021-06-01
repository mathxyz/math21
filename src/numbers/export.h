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

#include "config/config.h"

#if defined(MATH21_FLAG_USE_CUDA)

#include "config/gpu_config_c.h"

#endif

#if defined(MATH21_FLAG_USE_CPU)
#define M21_EXPORT
#define M21_EXPT_DEVICE
#elif defined(MATH21_FLAG_USE_CUDA)
#define M21_EXPORT __host__ __device__
#define M21_EXPT_DEVICE __device__
#elif defined(MATH21_FLAG_USE_OPENCL)
#define M21_EXPORT
#define M21_EXPT_DEVICE
#endif
