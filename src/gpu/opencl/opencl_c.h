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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_OPENCL

typedef struct m21dim2 {
    size_t x;
    size_t y;
} m21dim2;

extern int gpu_index;

void math21_opencl_set_device(int n);

void math21_opencl_checkError(cl_int error);

void math21_opencl_printPlatformInfoString(const char *name, cl_platform_id platformId, cl_platform_info info);

void math21_opencl_printPlatformInfo(const char *name, cl_platform_id platformId, cl_platform_info info);

int math21_opencl_getComputeUnits(cl_device_id device);

int math21_opencl_getLocalMemorySize(cl_device_id device);

int math21_opencl_getLocalMemorySizeKB(cl_device_id device);

int math21_opencl_getMaxWorkgroupSize(cl_device_id device);

int math21_opencl_getMaxAllocSizeMB(cl_device_id device);

m21dim2 math21_opencl_gridsize(size_t n);

void math21_opencl_push_array(m21clvector x_gpu, const float *x, size_t n);

void math21_opencl_push_N8_array(m21clvector x_gpu, const NumN8 *x, size_t n);

void math21_opencl_pull_array(m21clvector x_gpu, float *x, size_t n);

void math21_opencl_pull_N8_array(m21clvector x_gpu, NumN8 *x, size_t n);

void math21_opencl_vector_log_pointer(m21clvector x_gpu);

void math21_opencl_destroy();

#endif

#ifdef __cplusplus
}
#endif
