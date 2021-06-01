#define MATH21_OPENCL_BLOCK_SIZE 512

__kernel void math21_vector_loss_l1_opencl_kernel(int n,
                                                  __global const float *x, __global const float *t,
                                                  __global float *dx,
                                                  __global float *error) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) {
        float diff = t[i] - x[i];
        error[i] = fabs(diff);
        dx[i] = diff > 0 ? 1 : -1;
    }
}

__kernel void
math21_vector_loss_l2_opencl_kernel(int n, __global const float *x, __global const float *t, __global float *dx,
                                    __global float *error) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) {
        float diff = t[i] - x[i];
        error[i] = diff * diff;
        dx[i] = diff;
    }
}

__kernel void
math21_vector_loss_smooth_l1_opencl_kernel(int n, __global const float *x, __global const float *t, __global float *dx,
                                           __global float *error) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) {
        float diff = t[i] - x[i];
        float abs_val = fabs(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            dx[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            dx[i] = (diff > 0) ? 1 : -1;
        }
    }
}

__kernel void math21_vector_zero_by_thresh_opencl_kernel(int n, __global float *x, int stride_x, float thresh) {
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    size_t global_z = get_global_id(2);
    size_t i = global_z * get_global_size(0) * get_global_size(1)
               + global_y * get_global_size(0) + global_x;
    if (i < n) {
        if (fabs(x[i * stride_x]) < thresh) x[i * stride_x] = 0;
    }
}