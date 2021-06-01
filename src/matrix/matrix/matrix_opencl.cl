// Todo: use similar kernels in https://cnugteren.github.io/tutorial/pages/page4.html

#define MATH21_OPENCL_BLOCK_SIZE 512

// error
// C = k1*A*B + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_nn_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                __global const float *A, int lda,
                                                                __global const float *B, int ldb,
                                                                float k2,
                                                                __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int p = get_global_id(1);

    int t;
    float sum = 0.0f;
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {
        int k = p + t;
        sum += (p + t < n_common) ? A[i * lda + k] * B[k * ldb + j] : 0;
    }

    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        int index;
        sum = 0;
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];
    }

    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// error
// C = k1*A*B.t + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_nt_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                __global const float *A, int lda,
                                                                __global const float *B, int ldb,
                                                                float k2,
                                                                __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int p = get_global_id(1);

    int t;
    float sum = 0.0f;
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {
        int k = p + t;
        sum += (p + t < n_common) ? A[i * lda + k] * B[j * ldb + k] : 0;
    }

    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        int index;
        sum = 0;
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];
    }

    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// error
// C = k1*A.t*B + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_tn_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                __global const float *A, int lda,
                                                                __global const float *B, int ldb,
                                                                float k2,
                                                                __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int p = get_global_id(1);

    int t;
    float sum = 0.0f;
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {
        int k = p + t;
        sum += (p + t < n_common) ? A[k * lda + i] * B[k * ldb + j] : 0;
    }

    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        int index;
        sum = 0;
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];
    }

    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// error
// C = k1*A.t*B.t + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_tt_v2_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                __global const float *A, int lda,
                                                                __global const float *B, int ldb,
                                                                float k2,
                                                                __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int p = get_global_id(1);

    int t;
    float sum = 0.0f;
    for (t = 0; t < n_common; t += MATH21_OPENCL_BLOCK_SIZE) {
        int k = p + t;
        sum += (p + t < n_common) ? A[k * lda + i] * B[j * ldb + k] : 0;
    }

    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        int index;
        sum = 0;
        for (index = 0; index < MATH21_OPENCL_BLOCK_SIZE; ++index) sum += part[index];
    }

    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// C = k1*A*B + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_nn_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                   __global const float *A, int lda,
                                                                   __global const float *B, int ldb,
                                                                   float k2,
                                                                   __global float *C, int ldc) {
    int size = nr_C * nc_C;
    // z*dim(y) * dim(x) + y * dim(x) + x
    // x = blockIdx.x * blockDim.x + threadIdx.x
    // y = blockIdx.y * blockDim.y + threadIdx.y
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    int k;
    float sum = 0.0f;
    for (k = 0; k < n_common; ++k) {
        sum += A[i * lda + k] * B[k * ldb + j];
    }
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// C = k1*A*B.t + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_nt_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                   __global const float *A, int lda,
                                                                   __global const float *B, int ldb,
                                                                   float k2,
                                                                   __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    int k;
    float sum = 0.0f;
    for (k = 0; k < n_common; ++k) {
        sum += A[i * lda + k] * B[j * ldb + k];
    }
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// C = k1*A.t*B + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_tn_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                   __global const float *A, int lda,
                                                                   __global const float *B, int ldb,
                                                                   float k2,
                                                                   __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    int k;
    float sum = 0.0f;
    for (k = 0; k < n_common; ++k) {
        sum += A[k * lda + i] * B[k * ldb + j];
    }
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}

// C = k1*A.t*B.t + k2*C
__kernel void
math21_matrix_multiply_k1AB_add_k2C_similar_tt_naive_opencl_kernel(int nr_C, int nc_C, int n_common, float k1,
                                                                   __global const float *A, int lda,
                                                                   __global const float *B, int ldb,
                                                                   float k2,
                                                                   __global float *C, int ldc) {
    int size = nr_C * nc_C;
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int j = id % nc_C;
    id /= nc_C;
    int i = id % nr_C;

    int k;
    float sum = 0.0f;
    for (k = 0; k < n_common; ++k) {
        sum += A[k * lda + i] * B[j * ldb + k];
    }
    C[i * ldc + j] = k1 * sum + k2 * C[i * ldc + j];
}