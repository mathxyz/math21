//#define powf(X) pow(X)
//#define sqrtf(X) sqrt(X)
__kernel void
math21_optimization_adam_update_part_2_opencl_kernel(int x_size, __global float *x, __global float *m, __global float *v, float beta1, float beta2, float alpha, float eps, int t) {
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= x_size) return;

    // compute bias-corrected first moment estimate
    float mhat = m[i] / (1.f - pow(beta1, t));
    // compute bias-corrected second raw moment estimate
    float vhat = v[i] / (1.f - pow(beta2, t));

    // update
    // x = x - alpha * m / (sqrt(v) + eps)
    x[i] = x[i] + alpha * mhat / (sqrt(vhat) + eps);
}
