#define MATH21_OPENCL_BLOCK_SIZE 512
__kernel void math21_ml_batchnormalization_backward_mu_fast_opencl_kernel(__global const float *delta, __global const float *variance, int batch, int filters, int spatial, __global float *mean_delta)
{
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int id = get_global_id(1);
    int filter = get_global_id(0);
    float sum = 0;
    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE) {
            int index = j * spatial*filters + filter * spatial + i + id;
            sum += (i + id < spatial) ? delta[index] : 0;
        }
    }
    part[id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (id == 0) {
        mean_delta[filter] = 0;
        for (i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i) {
            mean_delta[filter] += part[i];
        }
        mean_delta[filter] *= (-1.f / sqrt(variance[filter] + .00001f));
    }
}
__kernel void math21_ml_batchnormalization_backward_mu_opencl_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));
}
__kernel void  math21_ml_batchnormalization_backward_sigma_square_fast_opencl_kernel(__global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta)
{
    __local float part[MATH21_OPENCL_BLOCK_SIZE];
    int id = get_global_id(1);
    int filter = get_global_id(0);
    float sum = 0;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += MATH21_OPENCL_BLOCK_SIZE){
            int index = j*spatial*filters + filter*spatial + i + id;
            sum += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }
    part[id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < MATH21_OPENCL_BLOCK_SIZE; ++i){
            variance_delta[filter] += part[i];
        }
        variance_delta[filter] *= -.5f * pow(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}
__kernel void math21_ml_batchnormalization_backward_input_opencl_kernel(int N, __global float *x, __global float *mean, __global float *variance, __global float *mean_delta, __global float *variance_delta, int batch, int filters, int spatial, __global float *delta)
{
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    int f = (index/spatial)%filters;
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

