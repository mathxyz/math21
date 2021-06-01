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
#include "conv_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;
const static std::string kernel_file = "conv_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

void math21_ml_function_conv_X_to_X_prime_opencl(PointerFloatInputWrapper X,
                                                 int nch_X, int nr_X, int nc_X,
                                                 int ksize, int stride, int pad, PointerFloatWrapper X_prime) {
    int height_col = (nr_X + 2 * pad - ksize) / stride + 1;
    int width_col = (nc_X + 2 * pad - ksize) / stride + 1;
    int num_kernels = nch_X * height_col * width_col;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_conv_X_to_X_prime_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &num_kernels));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &X.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void *) &nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void *) &ksize));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &pad));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void *) &stride));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &height_col));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void *) &width_col));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &X_prime.buffer));

//    size_t global_size[] = {(num_kernels + MATH21_OPENCL_BLOCK_SIZE - 1) / MATH21_OPENCL_BLOCK_SIZE,
//                            MATH21_OPENCL_BLOCK_SIZE};

    m21dim2 dim = math21_opencl_gridsize(num_kernels);
    size_t global_size[] = {dim.x, dim.y, MATH21_OPENCL_BLOCK_SIZE};

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0,
                                          NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void
math21_ml_function_conv_binarize_weights_opencl(PointerFloatInputWrapper weights, int features_size, int size, PointerFloatWrapper binary)
{
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_conv_binarize_weights_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&weights.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&features_size));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&size));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&binary.buffer));

    m21dim2 dim = math21_opencl_gridsize(features_size);
    size_t global_size[] = { dim.x,dim.y,MATH21_OPENCL_BLOCK_SIZE };

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_function_conv_binarize_input_opencl(PointerFloatInputWrapper input, int n, PointerFloatWrapper binary)
{
    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_conv_binarize_input_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&binary.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = { dim.x,dim.y,MATH21_OPENCL_BLOCK_SIZE };

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_function_conv_dX_prime_to_dX_opencl(PointerFloatInputWrapper dX_prime,
                                                   int nch_X, int nr_X, int nc_X,
                                                   int ksize, int stride, int pad, PointerFloatWrapper dX){
    int nc_X_prime_1 = (nr_X + 2 * pad - ksize) / stride + 1;
    int nc_X_prime_2 = (nc_X + 2 * pad - ksize) / stride + 1;
    int num_kernels = nch_X * nr_X * nc_X;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_conv_dX_prime_to_dX_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&num_kernels));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dX_prime.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&nr_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&nc_X));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&ksize));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&pad));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&stride));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&nc_X_prime_1));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(int), (void*)&nc_X_prime_2));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&dX.buffer));

    m21dim2 dim = math21_opencl_gridsize(num_kernels);
    size_t global_size[] = { dim.x,dim.y,MATH21_OPENCL_BLOCK_SIZE };

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

void math21_ml_function_conv_smooth_opencl(mlfunction_conv *l, int size, float rate)
{
    int h = l->out_h;
    int w = l->out_w;
    int c = l->out_c;

    size_t n = h * w*c*l->batch;

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel = math21_opencl_getKernel(program, "math21_ml_function_conv_smooth_opencl_kernel");

    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&l->output.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&l->w));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&l->h));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&l->c));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&size));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(float), (void*)&rate));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&l->delta.buffer));

    m21dim2 dim = math21_opencl_gridsize(n);
    size_t global_size[] = { dim.x,dim.y,MATH21_OPENCL_BLOCK_SIZE };

    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL, global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif