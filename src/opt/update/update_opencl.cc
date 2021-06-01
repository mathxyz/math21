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
#include "update_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "update_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_optimization_adam_update_part_2_opencl(int x_size, PointerFloatWrapper x, PointerFloatWrapper m, PointerFloatWrapper v, float beta1, float beta2, float alpha, float eps, int t){
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_optimization_adam_update_part_2_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &x_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &x.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &m.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &v.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &beta1));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(float), (void *) &beta2));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(float), (void *) &alpha));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(float), (void *) &eps));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void *) &t));

    m21dim2 dim = math21_opencl_gridsize(x_size);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);

    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif