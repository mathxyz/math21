void math21_ml_function_softmax_opencl_device(__global float *input, int n, float temp, int stride, __global float *output) {
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for (i = 0; i < n; ++i) {
        int val = input[i * stride];
        largest = (val > largest) ? val : largest;
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i * stride] /= sum;
    }
}

__kernel void
math21_ml_function_softmax_tree_opencl_kernel(__global float *input, int spatial, int batch, int stride, float temp, __global float *output,
                                              int groups, __constant int *group_size, __constant int *group_offset) {
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= spatial * batch * groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g] * spatial;
    int boff = b * stride;
    math21_ml_function_softmax_opencl_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}
__kernel void
math21_ml_function_softmax_opencl_kernel(__global float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride,
                                         float temp, __global float *output) {
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
             get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= batch * groups) return;
    int b = id / groups;
    int g = id % groups;
    math21_ml_function_softmax_opencl_device(input + b * batch_offset + g * group_offset, n, temp, stride,
                                             output + b * batch_offset + g * group_offset);
}
__kernel void
math21_ml_function_softmax_x_ent_opencl_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i < n) {
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t - p;
    }
}
