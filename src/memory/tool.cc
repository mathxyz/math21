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

#include <cstring>
#include "inner_cc.h"
#include "tool.h"

namespace math21 {

    void xjmalloc(void **p_data, size_t size);

    void xjcalloc(void **p_data, size_t nmemb, size_t size);

    void xjrealloc(void **p_data, void *ptr, size_t size);

    //Sets the first num bytes of the block of memory pointed by ptr to the specified value (interpreted as an unsigned char).
    void xjmemset(void *ptr, int value, size_t num);

    void xjmemcpy(void *dest, const void *src, size_t n);

    void xjfree(void *ptr);

    /////////////////////

    void xjmalloc(void **p_data, size_t size) {
        *p_data = malloc(size);
    }

    void xjcalloc(void **p_data, size_t nmemb, size_t size) {
        *p_data = calloc(nmemb, size);
    }

    void xjrealloc(void **p_data, void *ptr, size_t size) {
        *p_data = realloc(ptr, size);
    }

    void xjmemset(void *ptr, int value, size_t num) {
//        memset(ptr, value, num);
    }

    void xjmemcpy(void *dest, const void *src, size_t n) {
        memcpy(dest, src, n);
    }

    void xjfree(void *ptr) {
        free(ptr);
    }

    /////////////////////

    // create cpu buffer. may deprecate
    void math21_vector_malloc(void **p_data, size_t size) {
#ifdef MATH21_FLAG_USE_CUDA
        // device code can access this host memory
        cudaMallocHost(p_data, size);
#else
        xjmalloc(p_data, size);
//        xjmemset(*p_data, 0, size);
#endif
    }

    void math21_memory_memcpy(void *dest, const void *src, size_t n) {
#ifdef MATH21_FLAG_USE_CUDA
        math21_cuda_memcpy_host_to_host(dest, src, n);
#else
        xjmemcpy(dest, src, n);
#endif
    }

    void math21_memory_free(void *ptr) {
#ifdef MATH21_FLAG_USE_CUDA
        cudaFreeHost(ptr);
#else
        xjfree(ptr);
#endif
    }

#ifdef MATH21_FLAG_USE_CUDA

    void math21_cuda_malloc_device(void **p_data, size_t size) {
        cudaMalloc(p_data, size);
    }

    void math21_cuda_free_device(void *ptr) {
        cudaFree(ptr);
    }

    void math21_cuda_Free(void *ptr, void *error) {
        cudaError_t &e = *(cudaError_t *) error;
        e = cudaFree(ptr);
    }

    void math21_cuda_Memset(void *devPtr, int value, size_t count) {
        cudaMemset(devPtr, value, count);
    }

    void math21_cuda_memcpy(void *dst, const void *src, size_t count, void *kind) {
        cudaMemcpy(dst, src, count, *(cudaMemcpyKind *) kind);
    }

    void math21_cuda_memcpy_host_to_host(void *dst, const void *src, size_t count) {
        cudaMemcpyKind kind = cudaMemcpyHostToHost;
        cudaMemcpy(dst, src, count, kind);
    }

    void math21_cuda_memcpy_host_to_device(void *dst, const void *src, size_t count) {
        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
        cudaMemcpy(dst, src, count, kind);
    }

    void math21_cuda_memcpy_device_to_host(void *dst, const void *src, size_t count) {
        cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
        cudaMemcpy(dst, src, count, kind);
    }

    void math21_cuda_DeviceSynchronize() {
        cudaDeviceSynchronize();
    }

    void math21_cuda_EventCreate(void *event) {
        cudaEventCreate((cudaEvent_t *) event);
    }

    void math21_cuda_EventRecord(void *event, void *stream) {
        cudaEventRecord(*(cudaEvent_t *) event, *(cudaStream_t *) stream);
    }

    void math21_cuda_EventSynchronize(void *event) {
        cudaEventSynchronize(*(cudaEvent_t *) event);
    }

    void math21_cuda_EventElapsedTime(float *ms, void *start, void *end) {
        cudaEventElapsedTime(ms, *(cudaEvent_t *) start, *(cudaEvent_t *) end);
    }

#endif

}