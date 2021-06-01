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
#include "vector_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "vector_opencl_02.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_vector_loss_l1_opencl(int n, m21clvector pred, m21clvector truth, m21clvector delta, m21clvector error) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_loss_l1_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &pred.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &truth.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &delta.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &error.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_loss_l2_opencl(int n, m21clvector pred, m21clvector truth, m21clvector delta, m21clvector error) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_loss_l2_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &pred.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &truth.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &delta.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &error.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void
math21_vector_loss_smooth_l1_opencl(int n, m21clvector pred, m21clvector truth, m21clvector delta, m21clvector error) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_loss_smooth_l1_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &pred.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &truth.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &delta.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &error.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_vector_zero_by_thresh_opencl(int N, m21clvector X, int INCX, float ALPHA) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_vector_zero_by_thresh_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &INCX));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(float), (void *) &ALPHA));

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


#endif