#define expf(X) exp(X)
#define floorf(X) floor(X)

typedef enum{
    MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC, MATH21_FUNCTION_ACTIVATION_TYPE_RELU, MATH21_FUNCTION_ACTIVATION_TYPE_RELIE, MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR, MATH21_FUNCTION_ACTIVATION_TYPE_RAMP, MATH21_FUNCTION_ACTIVATION_TYPE_TANH, MATH21_FUNCTION_ACTIVATION_TYPE_PLSE, MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY, MATH21_FUNCTION_ACTIVATION_TYPE_ELU, MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY, MATH21_FUNCTION_ACTIVATION_TYPE_STAIR, MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN, MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN, MATH21_FUNCTION_ACTIVATION_TYPE_SELU
} MATH21_FUNCTION_ACTIVATION_TYPE;

float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
float leaky_relu_activate_kernel(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2);
    else return (x - n) + floorf(x/2);
}


float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}
float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}
float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
float ramp_gradient_kernel(float x){return (x>0)+.1f;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
float stair_gradient_kernel(float x)
{
    if (floorf(x) == x) return 0;
    return 1;
}

float math21_function_activation_value_at_opencl_kernel(float x, int a)
{
    switch(a){
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:
            return linear_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:
            return logistic_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:
            return loggy_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:
            return relu_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:
            return elu_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:
            return selu_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:
            return relie_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:
            return ramp_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:
            return leaky_relu_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:
            return tanh_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:
            return plse_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:
            return stair_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:
            return hardtan_activate_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

float math21_function_activation_gradient_opencl_kernel(float x, int a)
{
    switch(a){
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:
            return linear_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:
            return logistic_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:
            return loggy_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:
            return relu_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:
            return elu_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:
            return selu_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:
            return relie_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:
            return ramp_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:
            return leaky_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:
            return tanh_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:
            return plse_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:
            return stair_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:
            return hardtan_gradient_kernel(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__kernel void math21_function_activation_vector_opencl_kernel(__global float *x, int n, int a)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n) x[i] = math21_function_activation_value_at_opencl_kernel(x[i], a);
}

__kernel void math21_function_activation_gradient_vector_opencl_kernel(__global float *x, int n, int a, __global float *delta)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n) delta[i] *= math21_function_activation_gradient_opencl_kernel(x[i], a);
}