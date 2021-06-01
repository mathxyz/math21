__kernel void math21_ml_function_dropout_forward_opencl_kernel(__global const float *x, __global float *y, int size, __global const float *rand, float prob, float scale)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id < size) y[id] = (rand[id] < prob) ? 0 : scale * x[id];
}

__kernel void math21_ml_function_dropout_backward_opencl_kernel(__global const float *x, __global float *y, int size, __global const float *rand, float prob, float scale)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id < size) {
        if(rand[id] >= prob){
            y[id] += scale * x[id];
        }
    }
}
