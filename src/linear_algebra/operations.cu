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

#include "inner_src.h"
#include "AffineTransform.h"

namespace math21 {

    namespace detail {
        __global__ void math21_cuda_kernel_warp_image_using_indexes(NumR *a, NumR *b, NumR *index,
                                                                    int a_nr, int a_nc,
                                                                    int b_nr, int b_nc, int b_nch,
                                                                    int index_nr, int index_nc,
                                                                    NumR I1_a,
                                                                    NumR I1_b,
                                                                    NumB I1_include_a,
                                                                    NumB I1_include_b,
                                                                    NumR I2_a,
                                                                    NumR I2_b,
                                                                    NumB I2_include_a,
                                                                    NumB I2_include_b) {
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < b_nr && j < b_nc) {
                NumZ ii, jj;
                ii = (NumZ) index[i * index_nc + j];
                jj = (NumZ) index[index_nr * index_nc + i * index_nc + j];
                if (
                        (ii > I1_a || (ii == I1_a && I1_include_a))
                        && (ii < I1_b || (ii == I1_b && I1_include_b))
                        && (jj > I2_a || (jj == I2_a && I2_include_a))
                        && (jj < I2_b || (jj == I2_b && I2_include_b))
                        ) {
                    for (NumN k = 0; k < b_nch; ++k) {
                        b[k * b_nr * b_nc + i * b_nc + j] = a[k * a_nr * a_nc + (ii - 1) * a_nc + (jj - 1)];
                    }
                }
            }
        }

        void math21_geometry_warp_image_using_indexes_cuda(const TenR &A, TenR &B, const TenR &index,
                                                           const Interval2D &I) {
            MATH21_ASSERT(!B.isEmpty() && A.dims() == 3 && A.dims() == B.dims() && A.dim(1) == B.dim(1),
                          "" << A.logInfo("A") << B.logInfo("B"))

            NumN nr, nc;
            nr = B.dim(2);
            nc = B.dim(3);

            NumN A_size = A.volume();
            NumN B_size = B.volume();
            NumN index_size = index.volume();

            MATH21_ASSERT(A.isStandard());
            MATH21_ASSERT(B.isStandard());
            MATH21_ASSERT(index.isStandard());

            const NumR *A_data = math21_memory_tensor_data_address(A);
            NumR *B_data = math21_memory_tensor_data_address(B);
            const NumR *index_data = math21_memory_tensor_data_address(index);

            NumR *d_a, *d_b, *d_index;
            math21_cuda_malloc_device((void **) &d_a, sizeof(NumR) * A_size);
            math21_cuda_malloc_device((void **) &d_b, sizeof(NumR) * B_size);
            math21_cuda_malloc_device((void **) &d_index, sizeof(NumR) * index_size);

            math21_cuda_memcpy_host_to_device(d_a, A_data, sizeof(NumR) * A_size);
            math21_cuda_memcpy_host_to_device(d_b, B_data, sizeof(NumR) * B_size);
            math21_cuda_memcpy_host_to_device(d_index, index_data, sizeof(NumR) * index_size);

            unsigned int grid_rows = (unsigned int) (nr + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
            unsigned int grid_cols = (unsigned int) (nc + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
            dim3 dimGrid(grid_cols, grid_rows, 1);
            dim3 dimBlock(MATH21_GPU_BLOCK_SIZE, MATH21_GPU_BLOCK_SIZE, 1);

//        timer t;
//        t.start();

            math21_cuda_kernel_warp_image_using_indexes << < dimGrid, dimBlock >> > (
                    d_a, d_b, d_index,
                            A.dim(2), A.dim(3),
                            B.dim(2), B.dim(3), B.dim(1),
                            index.dim(2), index.dim(3),
                            I(1).left(),
                            I(1).right(),
                            I(1).isLeftClosed(),
                            I(1).isLeftClosed(),
                            I(2).left(),
                            I(2).right(),
                            I(2).isLeftClosed(),
                            I(2).isLeftClosed());

            math21_cuda_memcpy_device_to_host(B_data, d_b,
                                              sizeof(NumR) * B_size);
            math21_cuda_DeviceSynchronize();

            math21_cuda_free_device(d_a);
            math21_cuda_free_device(d_b);
            math21_cuda_free_device(d_index);

//        t.end();
//        printf("Time elapsed %f ms.\n\n", t.time());
        }

    }
}