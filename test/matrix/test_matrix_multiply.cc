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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include "files.h"
#include "inner.h"

namespace math21 {
    void time_random_matrix(int TA, int TB, int m, int k, int n) {
        float *a;
        if (!TA) a = math21_matrix_create_random_cpu(m, k);
        else a = math21_matrix_create_random_cpu(k, m);
        int lda = (!TA) ? k : m;
        float *b;
        if (!TB) b = math21_matrix_create_random_cpu(k, n);
        else b = math21_matrix_create_random_cpu(n, k);
        int ldb = (!TB) ? n : k;

        float *c = math21_matrix_create_random_cpu(m, n);
        int i;
        clock_t start = clock(), end;
        for (i = 0; i < 10; ++i) {
            math21_matrix_multiply_k1AB_add_k2C_similar_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
        }
        end = clock();
        printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB,
               (float) (end - start) / CLOCKS_PER_SEC);
        math21_vector_free_cpu(a);
        math21_vector_free_cpu(b);
        math21_vector_free_cpu(c);
    }


    void time_gpu_random_matrix(int TA, int TB, int m, int k, int n) {
        float *a;
        if (!TA) a = math21_matrix_create_random_cpu(m, k);
        else a = math21_matrix_create_random_cpu(k, m);
        int lda = (!TA) ? k : m;
        float *b;
        if (!TB) b = math21_matrix_create_random_cpu(k, n);
        else b = math21_matrix_create_random_cpu(n, k);
        int ldb = (!TB) ? n : k;

        float *c = math21_matrix_create_random_cpu(m, n);
        int i;
        clock_t start = clock(), end;
        PointerFloatWrapper gpu_a = math21_vector_create_from_cpuvector_wrapper(m * k, a, 1);
        PointerFloatWrapper gpu_b = math21_vector_create_from_cpuvector_wrapper(n * k, b, 1);
        PointerFloatWrapper gpu_c = math21_vector_create_from_cpuvector_wrapper(m * n, c, 1);

        for (i = 0; i < 32; ++i) {
            math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(TA, TB, m, n, k, 1, gpu_a, lda, gpu_b, ldb, 1, gpu_c,
                                                                n);
        }
        math21_vector_pull_wrapper(gpu_c, c, m * n);
        end = clock();
        printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n", m, k, k, n, TA, TB,
               (float) (end - start) / CLOCKS_PER_SEC);
        math21_vector_free_cpu(a);
        math21_vector_free_cpu(b);
        math21_vector_free_cpu(c);
        math21_vector_free_wrapper(gpu_a);
        math21_vector_free_wrapper(gpu_b);
        math21_vector_free_wrapper(gpu_c);
    }

    void time_gpu(int TA, int TB, int m, int k, int n) {
        int iter = 10;
        float *a = math21_matrix_create_random_cpu(m, k);
        float *b = math21_matrix_create_random_cpu(k, n);

        int lda = (!TA) ? k : m;
        int ldb = (!TB) ? n : k;

        float *c = math21_matrix_create_random_cpu(m, n);

        int i;
        timer time;
        time.start();
        PointerFloatWrapper gpu_a = math21_vector_create_from_cpuvector_wrapper(m * k, a, 1);
        PointerFloatWrapper gpu_b = math21_vector_create_from_cpuvector_wrapper(n * k, b, 1);
        PointerFloatWrapper gpu_c = math21_vector_create_from_cpuvector_wrapper(m * n, c, 1);


        for (i = 0; i < iter; ++i) {
            math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(TA, TB, m, n, k, 1, gpu_a, lda, gpu_b, ldb, 1, gpu_c,
                                                                n);
        }
        double flop = ((double) m) * n * (2. * k + 2.) * iter;
        double gflop = flop / pow(10., 9);
        math21_vector_pull_wrapper(gpu_c, c, m * n);
        time.end();
        printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n", m, k, k, n, TA, TB, (time.time()*1e-3),
               gflop / (time.time()*1e-3));
        math21_vector_free_cpu(a);
        math21_vector_free_cpu(b);
        math21_vector_free_cpu(c);
        math21_vector_free_wrapper(gpu_a);
        math21_vector_free_wrapper(gpu_b);
        math21_vector_free_wrapper(gpu_c);
    }

    void test_gpu_accuracy(int TA, int TB, int m, int k, int n) {
        math21_c_seed_random_generator(212121);
        float *a;
        if (!TA) a = math21_matrix_create_random_cpu(m, k);
        else a = math21_matrix_create_random_cpu(k, m);
        int lda = (!TA) ? k : m;
        float *b;
        if (!TB) b = math21_matrix_create_random_cpu(k, n);
        else b = math21_matrix_create_random_cpu(n, k);
        int ldb = (!TB) ? n : k;

        float *c = math21_matrix_create_random_cpu(m, n);
        float *c_gpu = math21_matrix_create_random_cpu(m, n);
        memset(c, 0, m * n * sizeof(float));
        memset(c_gpu, 0, m * n * sizeof(float));

        PointerFloatWrapper gpu_a = math21_vector_create_from_cpuvector_wrapper(m * k, a, 1);
        PointerFloatWrapper gpu_b = math21_vector_create_from_cpuvector_wrapper(n * k, b, 1);
        PointerFloatWrapper gpu_c = math21_vector_create_from_cpuvector_wrapper(m * n, c_gpu, 1);

        int i;
        //pm(m,k,b);
        math21_matrix_multiply_k1AB_add_k2C_similar_wrapper(TA, TB, m, n, k, 1, gpu_a, lda, gpu_b, ldb, 1, gpu_c, n);
        math21_vector_pull_wrapper(gpu_c, c_gpu, m * n);
        //printf("GPU\n");
        //pm(m, n, c_gpu);

        math21_matrix_multiply_k1AB_add_k2C_similar_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
        //printf("\n\nCPU\n");
        //pm(m, n, c);
        double sse = 0;
        for (i = 0; i < m * n; ++i) {
            //printf("%f %f\n", c[i], c_gpu[i]);
            sse += pow(c[i] - c_gpu[i], 2);
        }
        printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n", m, k, k, n, TA, TB, sse / (m * n));
        math21_vector_free_cpu(a);
        math21_vector_free_cpu(b);
        math21_vector_free_cpu(c);
        math21_vector_free_cpu(c_gpu);
        math21_vector_free_wrapper(gpu_a);
        math21_vector_free_wrapper(gpu_b);
        math21_vector_free_wrapper(gpu_c);
    }

    int test_gpu_matrix_multiply() {

        test_gpu_accuracy(0, 0, 10, 576, 75);

        test_gpu_accuracy(0, 0, 17, 10, 10);
        test_gpu_accuracy(1, 0, 17, 10, 10);
        test_gpu_accuracy(0, 1, 17, 10, 10);
        test_gpu_accuracy(1, 1, 17, 10, 10);

        test_gpu_accuracy(0, 0, 1000, 10, 100);
        test_gpu_accuracy(1, 0, 1000, 10, 100);
        test_gpu_accuracy(0, 1, 1000, 10, 100);
        test_gpu_accuracy(1, 1, 1000, 10, 100);

        test_gpu_accuracy(0, 0, 10, 10, 10);

//        time_gpu(0, 0, 64, 2916, 363);
//        time_gpu(0, 0, 64, 2916, 363);
//        time_gpu(0, 0, 64, 2916, 363);
//        time_gpu(0, 0, 192, 729, 1600);
//        time_gpu(0, 0, 384, 196, 1728);
//        time_gpu(0, 0, 256, 196, 3456);
//        time_gpu(0, 0, 256, 196, 2304);
//       time_gpu(0,0,128,4096,12544);
//       time_gpu(0,0,128,4096,4096);

//    time_gpu(0, 0, 64, 75, 12544);
//    time_gpu(0, 0, 64, 75, 12544);
//    time_gpu(0, 0, 64, 75, 12544);
//    time_gpu(0, 0, 64, 576, 12544);
//    time_gpu(0, 0, 256, 2304, 784);
//    time_gpu(1, 1, 2304, 256, 784);
//    time_gpu(0, 0, 512, 4608, 196);
//    time_gpu(1, 1, 4608, 512, 196);

        return 0;
    }
}