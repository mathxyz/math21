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

#include "inner.h"
#include "opencl_c.h"
#include "clvector.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

// offset can be negative.
m21clvector math21_opencl_vector_pointer_increase(m21clvector buffer, int offset_relative) {

    MATH21_ASSERT(buffer.address != nullptr);
    MATH21_ASSERT(buffer.unit > 0);

    cl_int err;
    cl_buffer_region region;
    m21clvector clvector;

    int offset = buffer.offset + offset_relative;
//    MATH21_ASSERT(offset >= 0, "offset = " << offset << "\n"
//                                           << "parent origin = " << buffer.offset << "\n"
//                                           << "offset_relative = " << offset_relative);

    int size_sub_buffer = (int) buffer.size_address - offset;
    if (offset >= 0 && size_sub_buffer > 0) {
        region.origin = offset * buffer.unit;
        region.size = size_sub_buffer * buffer.unit;

        cl_mem sub = clCreateSubBuffer(buffer.address, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        math21_opencl_checkError(err);

        math21_opencl_vector_set(&clvector, (size_t) size_sub_buffer, sub, buffer.size_address, buffer.address, offset,
                                 buffer.unit);
    } else {
        math21_opencl_vector_set(&clvector, 0, nullptr, buffer.size_address, buffer.address, offset, buffer.unit);
    }
    return clvector;
}

m21clvector operator+(m21clvector buffer, size_t size) {
    return math21_opencl_vector_pointer_increase(buffer, size);
}

m21clvector operator-(m21clvector buffer, size_t size) {
    int offset = -1 * (int) size;
    return math21_opencl_vector_pointer_increase(buffer, offset);
}

//m21clvector &m21clvector::operator+=(size_t offset) {
//    *this = *this + offset;
//    return *this;
//}

m21clvector &operator+=(m21clvector &buffer, size_t size) {
    buffer = buffer + size;
    return buffer;
}

m21clvector &operator-=(m21clvector &buffer, size_t size) {
    buffer = buffer - size;
    return buffer;
}

bool operator==(const m21clvector &x, const m21clvector &y) {
    return x.size == y.size &&
           x.buffer == y.buffer &&
           x.size_address == y.size_address &&
           x.address == y.address &&
           x.offset == y.offset &&
           x.unit == y.unit;
}

bool operator!=(const m21clvector &x, const m21clvector &y) {
    return !(x == y);
}

NumB math21_opencl_vector_isEmpty(const m21clvector *clvector) {
    if (clvector->buffer && clvector->size > 0) {
        return 0;
    } else {
        return 1;
    }
}

void math21_opencl_vector_reset(m21clvector *clvector) {
    clvector->size = 0;
    clvector->buffer = 0;

    clvector->size_address = 0;
    clvector->address = 0;

    clvector->offset = 0;
    clvector->unit = 0;
};

void math21_opencl_vector_set(m21clvector *clvector, size_t size, cl_mem buffer, size_t size_address, cl_mem address,
                              int offset, int unit) {
    clvector->size = size;
    clvector->buffer = buffer;
    clvector->size_address = size_address;
    clvector->address = address;
    clvector->offset = offset;
    clvector->unit = unit;
};

void math21_opencl_vector_free(m21clvector x_gpu) {
    cl_int status = clReleaseMemObject(x_gpu.buffer);
    math21_opencl_checkError(status);

    // added by ye
//    status = clReleaseMemObject(x_gpu.buffer);
//    math21_opencl_checkError(status);
}

#endif