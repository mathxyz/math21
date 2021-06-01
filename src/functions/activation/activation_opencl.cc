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
#include "activation.h"
#include "activation_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "activation_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_function_activation_vector_opencl(PointerFloatWrapper x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a){
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_function_activation_vector_opencl_kernel");

    int activation = (int) a;

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &activation));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_function_activation_gradient_vector_opencl(PointerFloatInputWrapper y, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, PointerFloatWrapper dy){
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_function_activation_gradient_vector_opencl_kernel");

    int activation = (int) a;

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &y.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &activation));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &dy.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif
