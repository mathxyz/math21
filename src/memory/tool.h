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

    void math21_vector_malloc(void **p_data, size_t size);

    void math21_memory_memcpy(void *dest, const void *src, size_t n);

    void math21_memory_free(void *ptr);

#ifdef MATH21_FLAG_USE_CUDA

    void math21_cuda_malloc_device(void **p_data, size_t size);

    void math21_cuda_free_device(void *ptr);

    void math21_cuda_Free(void *ptr, void *error);

    void math21_cuda_Memset(void *devPtr, int value, size_t count);

    void math21_cuda_memcpy(void *dst, const void *src, size_t count, void* kind);

    void math21_cuda_memcpy_host_to_host(void *dst, const void *src, size_t count);

    void math21_cuda_memcpy_host_to_device(void *dst, const void *src, size_t count);

    void math21_cuda_memcpy_device_to_host(void *dst, const void *src, size_t count);

    void math21_cuda_DeviceSynchronize();

    void math21_cuda_EventCreate(void *event);

    void math21_cuda_EventRecord(void *event, void *stream);

    void math21_cuda_EventSynchronize(void *event);

    void math21_cuda_EventElapsedTime(float *ms, void *start, void *end);

#endif

}