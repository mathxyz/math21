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

#include <random>
#include "inner_cc.h"
#include "vector_cpu.h"
#include "vector_wrapper.h"
#include "vector_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "vector_opencl_01.cl";
static std::shared_ptr<m21clprogram> program = NULL;

m21clvector math21_vector_deserialize_c_opencl(FILE *f, size_t *n0) {
    float *x = math21_vector_deserialize_c_cpu(f, n0);
    size_t n = *n0;
    if (!x || n <= 0) {
        return math21_vector_getEmpty_R32_wrapper();
    }
    auto v = math21_vector_create_with_default_value_opencl(n, 0);
    math21_opencl_push_array(v, x, n);
    math21_vector_free_cpu(x);
    return v;
}

m21clvector math21_vector_deserialize_from_file_opencl(const char *name, size_t *n) {
    FILE *f = fopen(name, "rb");
    if (!f) {
        math21_file_error(name);
        return math21_vector_getEmpty_R32_wrapper();
    }
    auto v = math21_vector_deserialize_c_opencl(f, n);
    fclose(f);
    return v;
}

void math21_vector_serialize_c_opencl(FILE *f, m21clvector v, size_t n) {
    if (math21_vector_isEmpty_wrapper(v) || n <= 0) {
        return;
    }
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_opencl_pull_array(v, x, n);
    math21_vector_serialize_c_cpu(f, x, n);
    math21_vector_free_cpu(x);
}

void math21_vector_serialize_to_file_opencl(const char *name, m21clvector v, size_t n) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        math21_file_error(name);
        return;
    }
    math21_vector_serialize_c_opencl(f, v, n);
    fclose(f);
}

// [from, to)
void math21_vector_save_opencl(const char *name, m21clvector v, size_t from, size_t to) {
    size_t n = to - from;
    if (math21_vector_isEmpty_wrapper(v) || n <= 0) {
        return;
    }
    v += from;
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_opencl_pull_array(v, x, n);
    math21_vector_save_cpu(name, x, 0, n);
    math21_vector_free_cpu(x);
}

// [from, to)
void math21_vector_log_opencl(const char *name, m21clvector v, size_t from, size_t to) {
    size_t n = to - from;
    v += from;
    float *x = math21_vector_create_with_default_value_cpu(n, 0);
    math21_opencl_pull_array(v, x, n);
    math21_vector_log_cpu(name, x, 0, n);
    math21_vector_free_cpu(x);
}

m21clvector math21_vector_create_with_default_value_opencl(size_t n, float value) {
    cl_mem x_gpu;
    cl_int error;
    size_t size = sizeof(float) * n;
    x_gpu = clCreateBuffer(math21_opencl_get_context(), CL_MEM_READ_WRITE, size, 0, &error);
    math21_opencl_checkError(error);
    m21clvector clvector;
    math21_opencl_vector_set(&clvector, n, x_gpu, n, x_gpu, 0, sizeof(float));
    math21_vector_set_wrapper(n, value, clvector, 1);
    if (!x_gpu) MATH21_ASSERT(0, "opencl malloc failed\n");
    return clvector;
}

m21clvector math21_vector_create_with_default_value_int_opencl(size_t n, int value) {
    cl_mem x_gpu;
    cl_int error;
    size_t size = sizeof(int) * n;
    x_gpu = clCreateBuffer(math21_opencl_get_context(), CL_MEM_READ_WRITE, size, 0, &error);
    math21_opencl_checkError(error);
    m21clvector clvector;
    math21_opencl_vector_set(&clvector, n, x_gpu, n, x_gpu, 0, sizeof(int));
    math21_vector_set_int_wrapper(n, value, clvector, 1);
    if (!x_gpu) MATH21_ASSERT(0, "opencl malloc failed\n");
    return clvector;
}

m21clvector math21_vector_create_buffer_opencl(size_t n, size_t elementSize) {
    MATH21_ASSERT(elementSize==1, "need considering\n");
    cl_mem x_gpu;
    cl_int error;
    size_t size = elementSize * n;
    x_gpu = clCreateBuffer(math21_opencl_get_context(), CL_MEM_READ_WRITE, size, 0, &error);
    math21_opencl_checkError(error);
    m21clvector clvector;
    math21_opencl_vector_set(&clvector, n, x_gpu, n, x_gpu, 0, sizeof(NumN8));
    if (!x_gpu) MATH21_ASSERT(0, "opencl malloc failed\n");
    return clvector;
}

PointerFloatWrapper math21_vector_resize_with_default_value_opencl(PointerFloatWrapper v, size_t n, float value) {
    math21_vector_free_opencl(v);
    v = math21_vector_create_with_default_value_opencl(n, value);
    return v;
}

PointerVoidWrapper math21_vector_resize_buffer_opencl(PointerVoidWrapper v, size_t n, size_t elementSize) {
    math21_vector_free_opencl(v);
    v = math21_vector_create_buffer_opencl(n, elementSize);
    return v;
}

PointerIntWrapper math21_vector_resize_with_default_value_int_opencl(PointerIntWrapper v, size_t n, int value) {
    math21_vector_free_opencl(v);
    v = math21_vector_create_with_default_value_int_opencl(n, value);
    return v;
}

m21clvector math21_vector_create_from_cpuvector_opencl(size_t n, const float *x, int stride_x) {
    cl_mem x_gpu;
    cl_int error;
    size_t size = sizeof(float) * n;
    x_gpu = clCreateBuffer(math21_opencl_get_context(), CL_MEM_READ_WRITE, size, 0, &error);
    math21_opencl_checkError(error);

    m21clvector clvector;
    math21_opencl_vector_set(&clvector, n, x_gpu, n, x_gpu, 0, sizeof(float));

    if (!x) {
        math21_vector_set_wrapper(n, 0, clvector, 1);
    } else {
        if (stride_x == 1) {
            error = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu, CL_TRUE, 0, size, x, 0, NULL, NULL);
            math21_opencl_checkError(error);
        } else {
            float *v = math21_vector_create_from_cpuvector_cpu(n, x, stride_x);
            error = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu, CL_TRUE, 0, size, v, 0, NULL, NULL);
            math21_opencl_checkError(error);
            math21_vector_free_cpu(v);
        }
    }
    if (!x_gpu) MATH21_ASSERT(0, "opencl malloc failed\n");
    return clvector;
}

m21clvector math21_vector_create_from_cpuvector_int_opencl(size_t n, const int *x, int stride_x) {
    cl_mem x_gpu;
    cl_int error;
    size_t size = sizeof(int) * n;
    x_gpu = clCreateBuffer(math21_opencl_get_context(), CL_MEM_READ_WRITE, size, 0, &error);
    math21_opencl_checkError(error);

    m21clvector clvector;
    math21_opencl_vector_set(&clvector, n, x_gpu, n, x_gpu, 0, sizeof(int));

    if (!x) {
        math21_vector_set_int_wrapper(n, 0, clvector, 1);
    } else {
        if (stride_x == 1) {
            error = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu, CL_TRUE, 0, size, x, 0, NULL, NULL);
            math21_opencl_checkError(error);
        } else {
            auto *v = math21_vector_create_from_cpuvector_int_cpu(n, x, stride_x);
            error = clEnqueueWriteBuffer(math21_opencl_get_command_queue(), x_gpu, CL_TRUE, 0, size, v, 0, NULL, NULL);
            math21_opencl_checkError(error);
            math21_vector_free_cpu(v);
        }
    }
    if (!x_gpu) MATH21_ASSERT(0, "opencl malloc failed\n");
    return clvector;
}

void math21_vector_free_opencl(m21clvector x_gpu) {
    math21_opencl_vector_free(x_gpu);
}

void
math21_vector_mean_fast_opencl(PointerFloatInputWrapper X, int mini_batch_size, int features_size, int in_class_size,
                               PointerFloatWrapper mean) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_mean_fast_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &mean.buffer));

    size_t global_size[] = {static_cast<size_t>(features_size), MATH21_OPENCL_BLOCK_SIZE};
    size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size, group_size,
                                          0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_mean_opencl(PointerFloatInputWrapper X, int mini_batch_size, int features_size, int in_class_size,
                               PointerFloatWrapper mean) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_mean_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &mean.buffer));

    m21dim2 dim = math21_opencl_gridsize(features_size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_variance_fast_opencl(m21clvector x, m21clvector mean, int batch, int filters, int spatial,
                                        m21clvector variance) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_variance_fast_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &mean.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &filters));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &spatial));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &variance.buffer));

    size_t global_size[] = {static_cast<size_t>(filters), MATH21_OPENCL_BLOCK_SIZE};
    size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size, group_size,
                                          0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_variance_opencl(m21clvector x, m21clvector mean, int batch, int filters, int spatial,
                                   m21clvector variance) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_variance_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &mean.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &filters));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &spatial));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &variance.buffer));

    m21dim2 dim = math21_opencl_gridsize(filters);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void
math21_vector_assign_from_vector_with_offset_opencl(int N, m21clvector X, int OFFX, int INCX, m21clvector Y, int OFFY,
                                                    int INCY) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_assign_from_vector_with_offset_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &OFFX));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &INCX));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &Y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &OFFY));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &INCY));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_assign_from_vector_N8_opencl(int N, m21clvector X, m21clvector Y) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_assign_from_vector_N8_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Y.buffer));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_assign_from_vector_opencl(int n, m21clvector X, int stride_x, m21clvector Y, int stride_y) {
    math21_vector_assign_from_vector_with_offset_opencl(n, X, 0, stride_x, Y, 0, stride_y);
}

void math21_vector_kx_opencl(int N, float ALPHA, m21clvector X, int INCX) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_kx_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &ALPHA));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &INCX));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_k_add_x_opencl(int N, float ALPHA, m21clvector X, int INCX) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_k_add_x_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &ALPHA));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &INCX));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_kx_add_y_with_offset_opencl(int N, float ALPHA, m21clvector X, int OFFX, int INCX, m21clvector Y,
                                               int OFFY, int INCY) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_kx_add_y_with_offset_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &ALPHA));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &OFFX));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &INCX));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &Y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &OFFY));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &INCY));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_kx_add_y_opencl(int N, float ALPHA, m21clvector X, int INCX, m21clvector Y, int INCY) {
    math21_vector_kx_add_y_with_offset_opencl(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void math21_vector_normalize_opencl(m21clvector x, m21clvector mean, m21clvector variance, int batch, int filters,
                                    int spatial) {
    size_t N = batch * filters * spatial;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_normalize_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mean.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &variance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &filters));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &spatial));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_kx_with_in_class_opencl(m21clvector output, m21clvector biases, int batch, int n, int size) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_kx_with_in_class_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &output.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &biases.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &size));

    size_t global_size[] = {MATH21_OPENCL_BLOCK_SIZE, static_cast<size_t>((size - 1) / MATH21_OPENCL_BLOCK_SIZE + 1),
                            static_cast<size_t>(n)};
    auto *events = new cl_event[batch];

    for (int i = 0; i < batch; i++) {
        size_t offset[] = {0, 0, static_cast<size_t>(i * n)};
        cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, offset, global_size, NULL,
                                              0, nullptr, &events[i]);
        math21_opencl_checkError(error);
    }

    math21_opencl_checkError(clWaitForEvents(batch, events));
    for (int i = 0; i < batch; i++)
        clReleaseEvent(events[i]);
    delete[] events;
}

void math21_vector_x_add_b_with_in_class_opencl(m21clvector output, m21clvector biases, int batch, int n, int size) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_x_add_b_with_in_class_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &output.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &biases.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &size));

    int num = n * size * batch;
    m21dim2 dim = math21_opencl_gridsize(num);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_sum_with_in_class_opencl(m21clvector bias_updates, m21clvector delta, int batch, int n, int size) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_event e;
    if (size == 1) {
        cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_sum_with_in_class_conn_opencl_kernel");

        math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bias_updates.buffer));
        math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &delta.buffer));
        math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
        math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &n));

        // by ye
//        int num = n * size * batch;
        int num = n;
        m21dim2 dim = math21_opencl_gridsize(num);
        size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};
        cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                              NULL, &e);
        math21_opencl_checkError(error);
    } else {
        cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_sum_with_in_class_opencl_kernel");

        math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bias_updates.buffer));
        math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &delta.buffer));
        math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
        math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &n));
        math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &size));

        size_t global_size[] = {static_cast<size_t>(n), MATH21_OPENCL_BLOCK_SIZE};
        size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
        cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size,
                                              group_size, 0, NULL, &e);
        math21_opencl_checkError(error);
    }
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void
math21_vector_sum_SchurProduct_with_in_class_opencl(m21clvector x_norm, m21clvector delta, int batch, int n, int size,
                                                    m21clvector scale_updates) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_sum_SchurProduct_with_in_class_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &x_norm.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &delta.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &size));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &scale_updates.buffer));

    size_t global_size[] = {static_cast<size_t>(n), MATH21_OPENCL_BLOCK_SIZE};
    size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size, group_size,
                                          0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_set_opencl(int N, float ALPHA, m21clvector X, int INCX) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_set_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &ALPHA));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &INCX));

    m21dim2 work_size = math21_opencl_gridsize(N);
    size_t global_size[] = {work_size.x, work_size.y, MATH21_OPENCL_BLOCK_SIZE};
    size_t offset[] = {0, 0, 0};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, offset, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);

    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_set_int_opencl(int N, int ALPHA, m21clvector X, int INCX) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_set_int_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &ALPHA));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &INCX));

    m21dim2 work_size = math21_opencl_gridsize(N);
    size_t global_size[] = {work_size.x, work_size.y, MATH21_OPENCL_BLOCK_SIZE};
    size_t offset[] = {0, 0, 0};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, offset, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);

    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_feature2d_add_2_opencl(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y) {
    int nch = math21_number_min_2_int(nch_X, nch_Y);
    int nr = math21_number_min_2_int(nr_X, nr_Y);
    int nc = math21_number_min_2_int(nc_X, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_y = (float) nc_Y / nc;

    int size = mini_batch_size * nch * nr * nc;
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_feature2d_add_2_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &size));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &nch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &nr));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &nc));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(float), (void *) &kx));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &nch_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void *) &nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(int), (void *) &nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(float), (void *) &stride_r_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(float), (void *) &stride_c_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 12, sizeof(float), (void *) &ky));
    math21_opencl_checkError(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *) &Y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 14, sizeof(int), (void *) &nch_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 15, sizeof(int), (void *) &nr_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 16, sizeof(int), (void *) &nc_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 17, sizeof(float), (void *) &stride_r_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 18, sizeof(float), (void *) &stride_c_y));

    m21dim2 dim = math21_opencl_gridsize(size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_feature2d_add_3_opencl(
        int mini_batch_size,
        float kx, PointerFloatInputWrapper X, int nch_X, int nr_X, int nc_X,
        float kx2, PointerFloatInputWrapper X2, int nch_X2, int nr_X2, int nc_X2,
        float ky, PointerFloatWrapper Y, int nch_Y, int nr_Y, int nc_Y) {
    int nch = math21_number_min_3_int(nch_X, nch_X2, nch_Y);
    int nr = math21_number_min_3_int(nr_X, nr_X2, nr_Y);
    int nc = math21_number_min_3_int(nc_X, nc_X2, nc_Y);

    float stride_r_x = (float) nr_X / nr;
    float stride_r_x2 = (float) nr_X2 / nr;
    float stride_r_y = (float) nr_Y / nr;
    float stride_c_x = (float) nc_X / nc;
    float stride_c_x2 = (float) nc_X2 / nc;
    float stride_c_y = (float) nc_Y / nc;

    int size = mini_batch_size * nch * nr * nc;
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_feature2d_add_3_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &size));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &nch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &nr));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &nc));

    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(float), (void *) &kx));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &nch_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void *) &nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(int), (void *) &nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(float), (void *) &stride_r_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 11, sizeof(float), (void *) &stride_c_x));

    math21_opencl_checkError(clSetKernelArg(kernel, 12, sizeof(float), (void *) &kx2));
    math21_opencl_checkError(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *) &X2.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 14, sizeof(int), (void *) &nch_X2));
    math21_opencl_checkError(clSetKernelArg(kernel, 15, sizeof(int), (void *) &nr_X2));
    math21_opencl_checkError(clSetKernelArg(kernel, 16, sizeof(int), (void *) &nc_X2));
    math21_opencl_checkError(clSetKernelArg(kernel, 17, sizeof(float), (void *) &stride_r_x2));
    math21_opencl_checkError(clSetKernelArg(kernel, 18, sizeof(float), (void *) &stride_c_x2));

    math21_opencl_checkError(clSetKernelArg(kernel, 19, sizeof(float), (void *) &ky));
    math21_opencl_checkError(clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 21, sizeof(int), (void *) &nch_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 22, sizeof(int), (void *) &nr_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 23, sizeof(int), (void *) &nc_Y));
    math21_opencl_checkError(clSetKernelArg(kernel, 24, sizeof(float), (void *) &stride_r_y));
    math21_opencl_checkError(clSetKernelArg(kernel, 25, sizeof(float), (void *) &stride_c_y));

    m21dim2 dim = math21_opencl_gridsize(size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_feature2d_sumdownsample_opencl(int mini_batch_size, m21clvector X, int nch_X, int nr_X, int nc_X,
                                                  int stride_X,
                                                  float k, m21clvector Y) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    auto kernel = math21_opencl_getKernel(program, "math21_vector_feature2d_sumdownsample_opencl_kernel");
    int n = mini_batch_size * nch_X * nr_X * nc_X;
    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &nch_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &stride_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(float), (void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &Y.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_feature2d_upsample_opencl(int mini_batch_size, PointerFloatWrapper X, int nch_X, int nr_X, int nc_X,
                                             int stride_X,
                                             int is_upsample, float k, PointerFloatWrapper Y) {
    MATH21_ASSERT(is_upsample);
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    auto kernel = math21_opencl_getKernel(program, "math21_vector_feature2d_upsample_opencl_kernel");
    size_t n = mini_batch_size * nch_X * nr_X * nc_X * stride_X * stride_X;
    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &nch_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &stride_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(float), (void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &Y.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

// X shape <= Y shape
void math21_vector_feature2d_sample_opencl(
        int mini_batch_size,
        PointerFloatWrapper X, int nch_X, int nr_X, int nc_X, int stride_X, int is_upsample, float k,
        PointerFloatWrapper Y) {
    if (is_upsample) {
        math21_vector_feature2d_upsample_opencl(mini_batch_size, X, nch_X, nr_X, nc_X, stride_X,
                                                is_upsample, k, Y);
    } else {
        math21_vector_feature2d_sumdownsample_opencl(mini_batch_size, X, nch_X, nr_X, nc_X, stride_X,
                                                     k, Y);
    }
}

void math21_vector_clip_opencl(int n, float k, PointerFloatWrapper x, int stride) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_clip_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &k));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &stride));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_xy_opencl(int n, PointerFloatInputWrapper x, int stride_x, PointerFloatWrapper y, int stride_y) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_xy_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &stride_x));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &stride_y));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_assign_by_mask_opencl(int n, PointerFloatWrapper x, float mask_num, PointerFloatInputWrapper mask,
                                         float val) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_assign_by_mask_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(float), (void *) &mask_num));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &mask.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &val));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_kx_by_mask_opencl(int N, float scale, m21clvector X, m21clvector mask, float mask_num) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_kx_by_mask_opencl_kernel");
    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(float), (void *) &scale));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &mask.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &mask_num));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

// todo: compare with cuda version
// todo: may use cpu version and remove the following.
void math21_vector_pr_rand_uniform_01_opencl(m21clvector x_gpu, size_t n) {
    std::default_random_engine e;
    e.seed(math21_c_seed_get()); // by cl
    std::uniform_real_distribution<float> u(0.0, 1.0);
    float *buffer = new float[n];
    for (int i = 0; i < n; i++)
        buffer[i] = u(e);
    math21_opencl_push_array(x_gpu, buffer, n);
    delete[] buffer;
}

#endif