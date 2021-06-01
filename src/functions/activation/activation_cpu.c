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
#include "activation_cpu.h"

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
// todo: add slope alpha.
static inline float leaky_relu_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float y){return 1;}
static inline float logistic_gradient(float y){return (1-y)*y;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

float math21_function_activation_value_cpu(float x, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    switch(a){
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:
            return linear_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:
            return logistic_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:
            return loggy_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:
            return relu_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:
            return elu_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:
            return selu_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:
            return relie_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:
            return ramp_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:
            return leaky_relu_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:
            return tanh_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:
            return plse_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:
            return stair_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:
            return hardtan_activate(x);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

// Y = h(X)
void math21_function_activation_vector_cpu(float *x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = math21_function_activation_value_cpu(x[i], a);
    }
}

float math21_function_activation_gradient_cpu(float y, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    switch(a){
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:
            return linear_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:
            return logistic_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:
            return loggy_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:
            return relu_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:
            return elu_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:
            return selu_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:
            return relie_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:
            return ramp_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:
            return leaky_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:
            return tanh_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:
            return plse_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:
            return stair_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:
            return hardtan_gradient(y);
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:
            return lhtan_gradient(y);
    }
    return 0;
}

// dL/dx = dL/dy *.ele f.d(x)
void math21_function_activation_gradient_vector_cpu(const float *y, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, float *dy)
{
    int i;
    for(i = 0; i < n; ++i){
        dy[i] *= math21_function_activation_gradient_cpu(y[i], a);
    }
}
