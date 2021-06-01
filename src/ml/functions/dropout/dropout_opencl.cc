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

#include "inner_cc.h"
#include "dropout_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "dropout_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_ml_function_dropout_forward_opencl(mlfunction_dropout *f, mlfunction_node *finput, int is_train) {
    if (!is_train) return;
    int size = f->inputs * f->batch;
    if (f->i_time_step==0) {
        math21_vector_pr_rand_uniform_01_wrapper(f->rand, size);
    }

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_dropout_forward_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &finput->y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &f->y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &f->rand.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &f->rate));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(float), (void *) &f->scale));

    m21dim2 dim = math21_opencl_gridsize(size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_function_dropout_backward_opencl(mlfunction_dropout *f, mlfunction_node *finput) {
    if (math21_vector_isEmpty_wrapper(finput->dy)) return;
    int size = f->inputs * f->batch;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_dropout_backward_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &f->dy.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &finput->dy.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &f->rand.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &f->rate));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(float), (void *) &f->scale));

    m21dim2 dim = math21_opencl_gridsize(size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif