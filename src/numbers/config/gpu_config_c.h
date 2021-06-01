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

#include "config.h"

///////////////////////////////// CUDA ////////////////////////////////////
#ifdef MATH21_FLAG_USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <cublas_v2.h>

#endif

///////////////////////////////// OPENCL ////////////////////////////////////

#ifdef MATH21_FLAG_USE_OPENCL

#include<CL/cl.h>

/*
// put to device code
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
*/

#endif

#ifdef MATH21_FLAG_USE_OPENCL_BLAS

#include<clBLAS.h>

#endif
