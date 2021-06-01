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
#include "softmax_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "softmax_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_ml_function_softmax_tree_opencl(m21clvector input, int in_class_size, int mini_batch_size, int stride, float temp, m21clvector output, m21tree hier) {
    m21clvector tree_groups_size = math21_vector_create_from_cpuvector_int_wrapper(hier.groups, hier.group_size, 1);
    m21clvector tree_groups_offset = math21_vector_create_from_cpuvector_int_wrapper(hier.groups, hier.group_offset, 1);

    int num = in_class_size * mini_batch_size * hier.groups;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_softmax_tree_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &input.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &stride));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(float), (void *) &temp));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &output.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &hier.groups));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &tree_groups_size.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &tree_groups_offset.buffer));

    m21dim2 dim = math21_opencl_gridsize(num);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
    math21_vector_free_wrapper(tree_groups_size);
    math21_vector_free_wrapper(tree_groups_offset);
}

void math21_ml_function_softmax_opencl(PointerFloatWrapper input, int n, int mini_batch_size, int batch_offset, int groups, int group_offset, int stride, float temp, PointerFloatWrapper output){
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_softmax_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &input.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &n));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &batch_offset));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &groups));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &group_offset));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &stride));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(float), (void *) &temp));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &output.buffer));

    m21dim2 dim = math21_opencl_gridsize(mini_batch_size * groups);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int status = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                           NULL, &e);
    math21_opencl_checkError(status);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_function_softmax_x_ent_opencl(int n, PointerFloatWrapper pred, PointerFloatWrapper truth, PointerFloatWrapper delta, PointerFloatWrapper error){
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_softmax_x_ent_opencl_kernel");
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
#endif