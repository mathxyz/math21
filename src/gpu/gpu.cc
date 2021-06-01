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

#include <iostream>
#include "cuda_info.h"
#include "test_c.h"
#include "cuda/files_c.h"
#include "opencl/files_c.h"
#include "gpu_c.h"
#include "gpu.h"

#ifndef MATH21_FLAG_USE_CUDA

void math21_cuda_example_test() {}

#endif

namespace math21 {
    using namespace cuda;
#ifndef MATH21_FLAG_USE_CUDA

    void math21_cuda_version() {
        std::cout << "CUDA: Off" << std::endl;
    }

    void math21_cuda_hello_world() {}

    void math21_cuda_thrust_version() {}

    void math21_cuda_atomicAdd_test() {}

#endif

    void math21_cuda_test() {
        math21_cuda_version();
        math21_cuda_hello_world();
        math21_cuda_example_test();
        math21_cuda_thrust_version();
        m21log("CUDA Device Count", getCudaEnabledDeviceCount());
    }
}

using namespace math21;

NumB math21_gpu_is_available() {
#ifdef MATH21_FLAG_USE_CPU
    return 0;
#else
    return 1;
#endif
}

void math21_gpu_set_device_wrapper(int n) {
#if defined(MATH21_FLAG_USE_CPU)
    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_set_device(n);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_set_device(n);
#endif
}

void math21_gpu_set_device_by_global_variable_wrapper() {
#if defined(MATH21_FLAG_USE_CPU)
//    MATH21_ASSERT(0);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_cuda_set_device(m21CudaCurrentDevice);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_opencl_set_device(gpu_index);
#endif
}

void math21_gpu_set_global_variable_wrapper(int n) {
#if defined(MATH21_FLAG_USE_CPU)
#elif defined(MATH21_FLAG_USE_CUDA)
    m21CudaCurrentDevice = n;
#elif defined(MATH21_FLAG_USE_OPENCL)
    gpu_index = n;
#endif
}

NumZ math21_gpu_get_global_variable_wrapper() {
#if defined(MATH21_FLAG_USE_CPU)
    return -1;
#elif defined(MATH21_FLAG_USE_CUDA)
    return m21CudaCurrentDevice;
#elif defined(MATH21_FLAG_USE_OPENCL)
    return gpu_index;
#endif
}
