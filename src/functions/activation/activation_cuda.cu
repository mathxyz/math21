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

#include "inner_c.h"
#include "activation.h"
#include "activation_cuda.h"

__device__ float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}
__device__ float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
__device__ float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
__device__ float leaky_relu_activate_kernel(float x){return (x>0) ? x : .1f*x;}
__device__ float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
__device__ float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2);
    else return (x - n) + floorf(x/2);
}


__device__ float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
__device__ float linear_gradient_kernel(float x){return 1;}
__device__ float logistic_gradient_kernel(float x){return (1-x)*x;}
__device__ float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}
__device__ float relu_gradient_kernel(float x){return (x>0);}
__device__ float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
__device__ float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
__device__ float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
__device__ float ramp_gradient_kernel(float x){return (x>0)+.1f;}
__device__ float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
__device__ float tanh_gradient_kernel(float x){return 1-x*x;}
__device__ float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
__device__ float stair_gradient_kernel(float x)
{
    if (floorf(x) == x) return 0;
    return 1;
}

__device__ float math21_function_activation_value_at_cuda_kernel(float x, MATH21_FUNCTION_ACTIVATION_TYPE a)
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

__device__ float math21_function_activation_gradient_cuda_kernel(float x, MATH21_FUNCTION_ACTIVATION_TYPE a)
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

__global__ void math21_function_activation_vector_cuda_kernel(float *x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = math21_function_activation_value_at_cuda_kernel(x[i], a);
}

__global__ void math21_function_activation_gradient_vector_cuda_kernel(const float *x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= math21_function_activation_gradient_cuda_kernel(x[i], a);
}

void math21_function_activation_vector_cuda(float *x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    math21_function_activation_vector_cuda_kernel<<<math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE>>>(x, n, a);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void math21_function_activation_gradient_vector_cuda(const float *y, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, float *dy)
{
    math21_function_activation_gradient_vector_cuda_kernel<<<math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE>>>(y, n, a, dy);
    math21_cuda_check_error(cudaPeekAtLastError());
}