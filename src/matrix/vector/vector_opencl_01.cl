#define MATH21_OPENCL_BLOCK_SIZE 512

__kernel void math21_vector_mean_fast_opencl_kernel(__global const float *x, int batch, int filters, int spatial,
                                                    __global float *mean) {
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int id = get_global_id(1);
    int filter = get_global_id(0);
    float sum = 0;
    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {
            int index = j * spatial * filters + filter * spatial + i + id;
            sum += (i + id < spatial) ? x[index] : 0;
        }
    }
    part[id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (id == 0) {
        mean[filter] = 0;
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {
            mean[filter] += part[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__kernel void
math21_vector_mean_opencl_kernel(__global const float *x, int batch, int filters, int spatial, __global float *mean) {
    float scale = 1.f / (batch * spatial);
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    int j, k;
    mean[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__kernel void
math21_vector_variance_fast_opencl_kernel(__global const float *x, __global float *mean, int batch, int filters,
                                          int spatial, __global float *variance) {
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int id = get_global_id(1);
    int filter = get_global_id(0);
    float sum = 0;
    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {
            int index = j * spatial * filters + filter * spatial + i + id;

            sum += (i + id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;
        }
    }
    part[id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (id == 0) {
        variance[filter] = 0;
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {
            variance[filter] += part[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

__kernel void
math21_vector_variance_opencl_kernel(__global const float *x, __global float *mean, int batch, int filters, int spatial,
                                     __global float *variance) {
    float scale = 1.f / (batch * spatial - 1);
    int j, k;
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    variance[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__kernel void
math21_vector_assign_from_vector_with_offset_opencl_kernel(int N, __global const float *X, int OFFX, int INCX,
                                                           __global float *Y, int OFFY, int INCY) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < N) Y[i * INCY + OFFY] = X[i * INCX + OFFX];
}

__kernel void
math21_vector_assign_from_vector_N8_opencl_kernel(int N, __global const unsigned char *X, __global unsigned char *Y) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < N) Y[i] = X[i];
}

__kernel void math21_vector_kx_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < N) X[i * INCX] *= ALPHA;
}

__kernel void math21_vector_k_add_x_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < N) X[i * INCX] += ALPHA;
}

__kernel void
math21_vector_kx_add_y_with_offset_opencl_kernel(int N, float ALPHA, __global const float *X, int OFFX, int INCX,
                                                 __global float *Y, int OFFY, int INCY) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
}

__kernel void
math21_vector_normalize_opencl_kernel(int N, __global float *x, __global float *mean, __global float *variance,
                                      int batch, int filters, int spatial) {
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    int f = (index / spatial) % filters;
    x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + .00001f));
}

__kernel void
math21_vector_kx_with_in_class_opencl_kernel(__global float *output, __global float *biases, int n, int size) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t filter = global_z - get_global_offset(2);
    size_t x_dim_size = get_global_size(0);
    size_t offset = x_dim_size * global_y + global_x;
    if (offset < size) output[global_z * size + offset] *= biases[filter];
}

__kernel void
math21_vector_x_add_b_with_in_class_opencl_kernel(__global float *output, __global const float *biases, int batch,
                                                  int n, int size) {
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= n * size * batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;
    output[(k * n + j) * size + i] += biases[j];
}

__kernel void
math21_vector_sum_with_in_class_conn_opencl_kernel(__global float *bias_updates, __global float *delta, int batch,
                                                   int n) {
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= n) return;
    int b;
    float sum = 0;
    for (b = 0; b < batch; ++b) {
        int i = b * n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

__kernel void
math21_vector_sum_with_in_class_opencl_kernel(__global float *bias_updates, __global float *delta, int batch, int n,
                                              int size) {
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int i, b;
    int filter = get_global_id(0);
    int p = get_global_id(1);
    float sum = 0;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < size; i += MATH21_OPENCL_BLOCK_SIZE) {
            int index = p + i + size * (filter + n * b);
            sum += (p + i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) bias_updates[filter] += part[i];
    }
}

__kernel void
math21_vector_sum_SchurProduct_with_in_class_opencl_kernel(__global float *x_norm, __global float *delta, int batch,
                                                           int n, int size, __global float *scale_updates) {
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int i, b;
    int filter = get_global_id(0);
    int p = get_global_id(1);
    float sum = 0;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < size; i += MATH21_OPENCL_BLOCK_SIZE) {
            int index = p + i + size * (filter + n * b);
            sum += (p + i < size) ? delta[index] * x_norm[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) scale_updates[filter] += part[i];
    }
}

__kernel void math21_vector_set_opencl_kernel(int N, float ALPHA, __global float *X, int INCX) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < N) X[i * INCX] = ALPHA;
}

__kernel void math21_vector_set_int_opencl_kernel(int N, int ALPHA, __global int *X, int INCX) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < N) X[i * INCX] = ALPHA;
}

__kernel void math21_vector_feature2d_add_2_opencl_kernel(int size, int mini_batch_size,
                                                          int nch, int nr, int nc,
                                                          float kx, __global const float *X, int nch_X, int nr_X,
                                                          int nc_X,
                                                          float stride_r_x, float stride_c_x,
                                                          float ky, __global float *Y, int nch_Y, int nr_Y, int nc_Y,
                                                          float stride_r_y, float stride_c_y) {
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int ic = id % nc;
    id /= nc;
    int ir = id % nr;
    id /= nr;
    int ich = id % nch;
    id /= nch;
    int imb = id % mini_batch_size;

    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);
    Y[index_Y] = kx * X[index_X] + ky * Y[index_Y];
}

__kernel void math21_vector_feature2d_add_3_opencl_kernel(int size, int mini_batch_size,
                                                          int nch, int nr, int nc,
                                                          float kx, __global const float *X, int nch_X, int nr_X,
                                                          int nc_X,
                                                          float stride_r_x, float stride_c_x,
                                                          float kx2, __global const float *X2, int nch_X2, int nr_X2,
                                                          int nc_X2,
                                                          float stride_r_x2, float stride_c_x2,
                                                          float ky, __global float *Y, int nch_Y, int nr_Y, int nc_Y,
                                                          float stride_r_y, float stride_c_y) {
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int ic = id % nc;
    id /= nc;
    int ir = id % nr;
    id /= nr;
    int ich = id % nch;
    id /= nch;
    int imb = id % mini_batch_size;

    // X(imb, ich, ir*stride_r_x, ic*stride_c_x)
    int index_X = ((imb * nch_X + ich) * nr_X + (int) (ir * stride_r_x)) * nc_X + (int) (ic * stride_c_x);
    // X2(imb, ich, ir*stride_r_x2, ic*stride_c_x2)
    int index_X2 = ((imb * nch_X2 + ich) * nr_X2 + (int) (ir * stride_r_x2)) * nc_X2 + (int) (ic * stride_c_x2);
    // Y(imb, ich, ir*stride_r_y, ic*stride_c_y)
    int index_Y = ((imb * nch_Y + ich) * nr_Y + (int) (ir * stride_r_y)) * nc_Y + (int) (ic * stride_c_y);
    Y[index_Y] = kx * X[index_X] + kx2 * X2[index_X2] + ky * Y[index_Y];
}

// X shape <= Y shape
__kernel void math21_vector_feature2d_sumdownsample_opencl_kernel(int n, int mini_batch_size,
                                                                  __global float *X, int nch_X, int nr_X, int nc_X,
                                                                  int stride_X, float k, __global const float *Y) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= n) return;

    int index_X = i;
    int ic_X = i % nc_X;
    i = i / nc_X;
    int ir_X = i % nr_X;
    i = i / nr_X;
    int ich_X = i % nch_X;
    i = i / nch_X;
    int imb_X = i % mini_batch_size;

    int ic_Y_abs = ic_X * stride_X;
    int ir_Y_abs = ir_X * stride_X;
    int ich_Y = ich_X;
    int imb_Y = imb_X;

    int nc_Y = nc_X * stride_X;
    int nr_Y = nr_X * stride_X;
    int nch_Y = nch_X;

    int ksize = stride_X;
    for (int ir_K = 0; ir_K < ksize; ++ir_K) {
        for (int ic_K = 0; ic_K < ksize; ++ic_K) {
            int ir_Y = ir_Y_abs + ir_K;
            int ic_Y = ic_Y_abs + ic_K;
            int index_Y = ((imb_Y * nch_Y + ich_Y) * nr_Y + ir_Y) * nc_Y + ic_Y;
            X[index_X] += k * Y[index_Y];
        }
    }
}

// only upsample
__kernel void math21_vector_feature2d_upsample_opencl_kernel(int n, int mini_batch_size,
                                                             __global float *X, int nch_X, int nr_X, int nc_X,
                                                             int stride_X, float k, __global float *Y) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= n) return;
    int index_Y = i;
    int ic_Y = i % (nc_X * stride_X);
    i = i / (nc_X * stride_X);
    int ir_Y = i % (nr_X * stride_X);
    i = i / (nr_X * stride_X);
    int ich_Y = i % nch_X;
    i = i / nch_X;
    int imb_Y = i % mini_batch_size;

    int ic_X = ic_Y / stride_X;
    int ir_X = ir_Y / stride_X;
    int ich_X = ich_Y;

    int index_X = imb_Y * nch_X * nr_X * nc_X + ich_X * nr_X * nc_X + ir_X * nc_X + ic_X;

    Y[index_Y] += k * X[index_X];
}

__kernel void math21_vector_clip_opencl_kernel(int n, float k, __global float *x, int stride_x) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) x[i * stride_x] = fmin(k, fmax(-k, x[i * stride_x]));
//    if (i < n) x[i * stride_x] = fminf(k, fmaxf(-k, x[i * stride_x]));
}

__kernel void
math21_vector_xy_opencl_kernel(int n, __global const float *x, int stride_x, __global float *y, int stride_y) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) y[i * stride_y] *= x[i * stride_x];
}

__kernel void
math21_vector_assign_by_mask_opencl_kernel(int n, __global float *x, float mask_num, __global const float *mask,
                                           float val) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n && mask[i] == mask_num) x[i] = val;
}

__kernel void
math21_vector_kx_by_mask_opencl_kernel(int n, float k, __global float *x, __global const float *mask, float mask_num) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n && mask[i] == mask_num) x[i] *= k;
}