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
#include "matrix_opencl.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

const static std::string kernel_file = "matrix_opencl.cl";
static std::shared_ptr<m21clprogram> program = NULL;

#ifdef MATH21_FLAG_USE_OPENCL_BLAS

void math21_matrix_multiply_k1AB_add_k2C_similar_opencl(int ta, int tb, int nr_C, int nc_C, int n_common, float k1,
                                                        m21clvector A, int lda,
                                                        m21clvector B, int ldb,
                                                        float k2,
                                                        m21clvector C, int ldc) {
    cl_event e;
    clblasTranspose transA = (ta ? clblasTrans : clblasNoTrans);
    clblasTranspose transB = (tb ? clblasTrans : clblasNoTrans);

    clblasSgemm(
            clblasColumnMajor,//Make column major the same with cublasSgemm
            transB, transA, nc_C, nr_C, n_common, k1, B.buffer, 0, ldb, A.buffer, 0, lda, k2,
            C.buffer, 0, ldc,
            1,//cl_uint numCommandQueues,
            math21_opencl_get_command_queue_pointer(),//cl_command_queue *commandQueues,
            0,//cl_uint numEventsInWaitList,
            NULL,//const cl_event *eventWaitList,
            &e//cl_event *events
    );
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif

#ifndef MATH21_FLAG_USE_OPENCL_BLAS
void math21_matrix_multiply_k1AB_add_k2C_similar_opencl(int ta, int tb, int nr_C, int nc_C, int n_common, float k1,
              m21clvector A, int lda,
              m21clvector B, int ldb,
              float k2,
              m21clvector C, int ldc) {

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel;
    if (!ta && !tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_nn_naive_opencl_kernel");
    else if (ta && !tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_tn_naive_opencl_kernel");
    else if (!ta && tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_nt_naive_opencl_kernel");
    else
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_tt_naive_opencl_kernel");


    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &nr_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &nc_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &n_common));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(float), (void *) &k1));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &lda));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &B.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &ldb));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(float), (void *) &k2));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &C.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(int), (void *) &ldc));

    int size = nr_C*nc_C;
    m21dim2 dim = math21_opencl_gridsize(size);
    size_t global_size[] = { dim.x,dim.y,MATH21_OPENCL_BLOCK_SIZE };
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 3, NULL,
                                          global_size, NULL, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}
#endif

// error
void math21_matrix_multiply_k1AB_add_k2C_similar_opencl_2(int ta, int tb, int nr_C, int nc_C, int n_common, float k1,
                                                          m21clvector A, int lda,
                                                          m21clvector B, int ldb,
                                                          float k2,
                                                          m21clvector C, int ldc) {

    if (program == NULL) {
        program = math21_opencl_build_program_from_file(kernel_file, "");
    }
    cl_kernel kernel;
    if (!ta && !tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_nn_v2_opencl_kernel");
    else if (ta && !tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_tn_v2_opencl_kernel");
    else if (!ta && tb)
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_nt_v2_opencl_kernel");
    else
        kernel = math21_opencl_getKernel(program, "math21_matrix_multiply_k1AB_add_k2C_similar_tt_v2_opencl_kernel");


    math21_opencl_checkError(clSetKernelArg(kernel, 0, sizeof(int), (void *) &nr_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 1, sizeof(int), (void *) &nc_C));
    math21_opencl_checkError(clSetKernelArg(kernel, 2, sizeof(int), (void *) &n_common));
    math21_opencl_checkError(clSetKernelArg(kernel, 3, sizeof(float), (void *) &k1));
    math21_opencl_checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 5, sizeof(int), (void *) &lda));
    math21_opencl_checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &B.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 7, sizeof(int), (void *) &ldb));
    math21_opencl_checkError(clSetKernelArg(kernel, 8, sizeof(float), (void *) &k2));
    math21_opencl_checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &C.buffer));
    math21_opencl_checkError(clSetKernelArg(kernel, 10, sizeof(int), (void *) &ldc));

    size_t size = nr_C * nc_C;
    size_t global_work_size[] = {size, MATH21_OPENCL_BLOCK_SIZE};
    size_t local_work_size[] = {1, MATH21_OPENCL_BLOCK_SIZE};
    cl_event e;
    cl_int error = clEnqueueNDRangeKernel(math21_opencl_get_command_queue(), kernel, 2, NULL, global_work_size,
                                          local_work_size, 0, NULL, &e);
    math21_opencl_checkError(error);
    math21_opencl_checkError(clWaitForEvents(1, &e));
    clReleaseEvent(e);
}

#endif