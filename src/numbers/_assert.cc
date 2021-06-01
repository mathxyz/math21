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

#include "config/cuda_config.h"
#include "_assert.h"

namespace math21 {
#ifdef MATH21_FLAG_USE_CUDA

    void _math21_assert_cuda(void *p_err, const char *file, int line,
                             const char *func) {
        cudaError_t &err = *(cudaError_t *) p_err;
        if (cudaSuccess != err) {
            _math21_assert_print(0, cudaGetErrorString(err), file, line, func);
        }
    }

#endif

}