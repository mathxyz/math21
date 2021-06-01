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
#include "batch_normalization.h"
#include "batch_normalization_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "batch_normalization_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void
math21_ml_batchnormalization_backward_mu_fast_opencl(PointerFloatInputWrapper dX_hat, PointerFloatInputWrapper variance,
                                                     int mini_batch_size, int features_size, int in_class_size,
                                                     PointerFloatWrapper dmu) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_batchnormalization_backward_mu_fast_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &dX_hat.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &variance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &dmu.buffer));

    size_t global_size[] = {static_cast<size_t>(features_size), MATH21_OPENCL_BLOCK_SIZE};
    size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size, group_size,
                                          0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void
math21_ml_batchnormalization_backward_mu_opencl(m21clvector delta, m21clvector variance, int batch, int filters,
                                                int spatial, m21clvector mean_delta) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_batchnormalization_backward_mu_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &delta.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &variance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &batch));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &filters));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &spatial));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &mean_delta.buffer));

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
math21_ml_batchnormalization_backward_sigma_square_fast_opencl(PointerFloatInputWrapper X,
                                                               PointerFloatInputWrapper dX_hat,
                                                               PointerFloatInputWrapper mu,
                                                               PointerFloatInputWrapper variance, int mini_batch_size,
                                                               int features_size, int in_class_size,
                                                               PointerFloatWrapper dvariance) {
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_batchnormalization_backward_sigma_square_fast_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &dX_hat.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mu.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &variance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &dvariance.buffer));

    size_t global_size[] = {static_cast<size_t>(features_size), MATH21_OPENCL_BLOCK_SIZE};
    size_t group_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_size, group_size,
                                          0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_batchnormalization_backward_input_opencl(PointerFloatInputWrapper X, PointerFloatInputWrapper mu, PointerFloatInputWrapper variance, PointerFloatInputWrapper dmu, PointerFloatInputWrapper dvariance, int mini_batch_size,
                                                        int features_size, int in_class_size, PointerFloatWrapper dX_hat){
    size_t N = mini_batch_size* features_size* in_class_size;
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_batchnormalization_backward_input_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &N));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mu.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &variance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &dmu.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &dvariance.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &mini_batch_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void *) &in_class_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &dX_hat.buffer));

    m21dim2 dim = math21_opencl_gridsize(N);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif