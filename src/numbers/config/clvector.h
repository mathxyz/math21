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

#include "gpu_config_c.h"
#include "../number_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATH21_FLAG_USE_OPENCL

typedef struct m21clvector m21clvector;

// `calloc` or `math21_vector_getEmpty_R32_wrapper` can set a vector to empty.
// `math21_vector_isEmpty_wrapper` can tell if a vector is empty.
// todo: delete buffer or address?
struct m21clvector {
    size_t size; //size space is available. sub-vector size
    cl_mem buffer; // point to sub-vector. call math21_opencl_vector_free to release cl_mem

    // private
    // todo: may remove
    size_t size_address;//new and delete address. whole vector size.
    cl_mem address; // point to actual created space, the whole vector. set only once.

    // offset can be negative. When negative, this vector can't be accessed, but can use operator +=.
    // the offset in whole vector. The place at whole vector from which sub-vector starts.
    int offset;
    int unit; // number of bytes per element, i.e., element size
};

#endif

#ifdef __cplusplus
}
#endif
