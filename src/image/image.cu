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

#include "inner_cu.h"
#include "image.h"

namespace math21{

        namespace detail {
            __global__ void math21_image_argb8888_to_tensor_cuda_kernel(NumN32 *data, NumR *m,
                                                                        NumN nr, NumN nc, NumN nch) {

                int i2 = blockIdx.y * blockDim.y + threadIdx.y;
                int i3 = blockIdx.x * blockDim.x + threadIdx.x;

                if (i2 < nr && i3 < nc) {
                    NumN32 a, r, g, b;
                    NumN32 value;
                    value = data[i2*nc+i3];

                    a = (value >> 24) & 0xff;
                    r = (value >> 16) & 0xff;
                    g = (value >> 8) & 0xff;
                    b = (value) & 0xff;
                    m[i2 * nc + i3] = r;
                    m[nr * nc + i2 * nc + i3] = g;
                    m[2 * nr * nc + i2 * nc + i3] = b;
                    if (nch == 4) {
                        m[3 * nr * nc + i2 * nc + i3] = a;
                    }
                }
            }

            void math21_image_argb8888_to_tensor_cuda(NumN32 *data, TenR &m, NumN nr, NumN nc, NumN nch) {
                MATH21_ASSERT(nch == 3 || nch == 4)
                if (m.isSameSize(nch, nr, nc) == 0) {
                    m.setSize(nch, nr, nc);
                }
//                NumN i1, i2, i3;
//                NumN32 a, r, g, b;
//                NumN32 value;

                NumN data_size = nr * nc;
                NumN m_size = m.volume();

                MATH21_ASSERT(m.isStandard());

                NumR *m_data = math21_memory_tensor_data_address(m);

                NumN32 *d_data;
                NumR *d_m;
                math21_cuda_malloc_device((void **) &d_data, sizeof(NumN32) * data_size);
                math21_cuda_malloc_device((void **) &d_m, sizeof(NumR) * m_size);

                math21_cuda_memcpy_host_to_device(d_data, data, sizeof(NumN32) * data_size);

                unsigned int grid_rows = (unsigned int) (nr + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
                unsigned int grid_cols = (unsigned int) (nc + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
                dim3 dimGrid(grid_cols, grid_rows, 1);
                dim3 dimBlock(MATH21_GPU_BLOCK_SIZE, MATH21_GPU_BLOCK_SIZE, 1);

                math21_image_argb8888_to_tensor_cuda_kernel << < dimGrid, dimBlock >> >
                                                                          (d_data, d_m, nr, nc, nch);

                math21_cuda_memcpy_device_to_host(m_data, d_m,
                                                  sizeof(NumR) * m_size);
                math21_cuda_DeviceSynchronize();

                math21_cuda_free_device(d_data);
                math21_cuda_free_device(d_m);
            }

            __global__ void math21_image_tensor_to_argb8888_cuda_kernel(NumN32 *data, const NumR *m,
                                                                        NumN nr, NumN nc, NumN nch) {

                int i2 = blockIdx.y * blockDim.y + threadIdx.y;
                int i3 = blockIdx.x * blockDim.x + threadIdx.x;

                NumN32 a, r, g, b;
                NumN32 value;
                NumR max = 255;
                NumR min = 0;
                if (i2 < nr && i3 < nc) {
                    NumR x;
                    x = m[i2 * nc + i3];
                    if (x > max) {
                        x = max;
                    } else if (x < min) {
                        x = min;
                    }
                    r = (NumN32) x;
                    x = m[nr * nc + i2 * nc + i3];
                    if (x > max) {
                        x = max;
                    } else if (x < min) {
                        x = min;
                    }
                    g = (NumN32) x;
                    x = m[2 * nr * nc + i2 * nc + i3];
                    if (x > max) {
                        x = max;
                    } else if (x < min) {
                        x = min;
                    }
                    b = (NumN32) x;

                    if (nch == 3) {
                        value = 0xff000000 | (r << 16) | (g << 8) | b;
                    } else {
                        x = m[3 * nr * nc + i2 * nc + i3];
                        if (x > max) {
                            x = max;
                        } else if (x < min) {
                            x = min;
                        }
                        a = (NumN32) x;
                        value = (a << 24) | (r << 16) | (g << 8) | b;
                    }
                    data[i2 * nc + i3] = value;
                }
            }

            void math21_image_tensor_to_argb8888_cuda(NumN32 *data, const TenR &m) {
                MATH21_ASSERT(m.dims() == 3 &&
                              (m.dim(1) == 3 || m.dim(1) == 4))

                NumN nr, nc, nch;
                nr = m.dim(2);
                nc = m.dim(3);
                nch = m.dim(1);

//                NumN i1, i2, i3;
//                NumN32 a, r, g, b;
//                NumN32 value;

                NumN data_size = nr * nc;
                NumN m_size = m.volume();

                MATH21_ASSERT(m.isStandard());

                const NumR *m_data = math21_memory_tensor_data_address(m);

                NumN32 *d_data;
                NumR *d_m;
                math21_cuda_malloc_device((void **) &d_data, sizeof(NumN32) * data_size);
                math21_cuda_malloc_device((void **) &d_m, sizeof(NumR) * m_size);

                math21_cuda_memcpy_host_to_device(d_m, m_data, sizeof(NumR) * m_size);

                unsigned int grid_rows = (unsigned int) (nr + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
                unsigned int grid_cols = (unsigned int) (nc + MATH21_GPU_BLOCK_SIZE - 1) / MATH21_GPU_BLOCK_SIZE;
                dim3 dimGrid(grid_cols, grid_rows, 1);
                dim3 dimBlock(MATH21_GPU_BLOCK_SIZE, MATH21_GPU_BLOCK_SIZE, 1);

                math21_image_tensor_to_argb8888_cuda_kernel << < dimGrid, dimBlock >> >
                                                                          (d_data, d_m, nr, nc, nch);

                math21_cuda_memcpy_device_to_host(data, d_data,
                                                  sizeof(NumN32) * data_size);
                math21_cuda_DeviceSynchronize();

                math21_cuda_free_device(d_data);
                math21_cuda_free_device(d_m);
            }
        }
}