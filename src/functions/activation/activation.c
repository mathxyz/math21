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

#include <string.h>
#include "activation.h"
#include "activation_cuda.h"
#include "activation_opencl.h"
#include "activation_cpu.h"

char *math21_function_activation_get_string(MATH21_FUNCTION_ACTIVATION_TYPE a)
{
    switch(a){
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC:
            return "logistic";
        case MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY:
            return "loggy";
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELU:
            return "relu";
        case MATH21_FUNCTION_ACTIVATION_TYPE_ELU:
            return "elu";
        case MATH21_FUNCTION_ACTIVATION_TYPE_SELU:
            return "selu";
        case MATH21_FUNCTION_ACTIVATION_TYPE_RELIE:
            return "relie";
        case MATH21_FUNCTION_ACTIVATION_TYPE_RAMP:
            return "ramp";
        case MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR:
            return "linear";
        case MATH21_FUNCTION_ACTIVATION_TYPE_TANH:
            return "tanh";
        case MATH21_FUNCTION_ACTIVATION_TYPE_PLSE:
            return "plse";
        case MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY:
            return "leaky";
        case MATH21_FUNCTION_ACTIVATION_TYPE_STAIR:
            return "stair";
        case MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN:
            return "hardtan";
        case MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

MATH21_FUNCTION_ACTIVATION_TYPE math21_function_activation_get_type(const char *s)
{
    if (strcmp(s, "logistic")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_LOGISTIC;
    if (strcmp(s, "loggy")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_LOGGY;
    if (strcmp(s, "relu")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_RELU;
    if (strcmp(s, "elu")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_ELU;
    if (strcmp(s, "selu")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_SELU;
    if (strcmp(s, "relie")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_RELIE;
    if (strcmp(s, "plse")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_PLSE;
    if (strcmp(s, "hardtan")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_HARDTAN;
    if (strcmp(s, "lhtan")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_LHTAN;
    if (strcmp(s, "linear")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_LINEAR;
    if (strcmp(s, "ramp")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_RAMP;
    if (strcmp(s, "leaky")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_LEAKY;
    if (strcmp(s, "tanh")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_TANH;
    if (strcmp(s, "stair")==0) return MATH21_FUNCTION_ACTIVATION_TYPE_STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return MATH21_FUNCTION_ACTIVATION_TYPE_RELU;
}

void math21_function_activation_vector_wrapper(PointerFloatWrapper x, int n, MATH21_FUNCTION_ACTIVATION_TYPE a)
{
#if defined(MATH21_FLAG_USE_CPU)
    math21_function_activation_vector_cpu(x, n, a);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_function_activation_vector_cuda(x, n, a);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_function_activation_vector_opencl(x, n, a);
#endif
}

void math21_function_activation_gradient_vector_wrapper(PointerFloatInputWrapper y, int n, MATH21_FUNCTION_ACTIVATION_TYPE a, PointerFloatWrapper dy)
{
#if defined(MATH21_FLAG_USE_CPU)
    math21_function_activation_gradient_vector_cpu(y, n, a, dy);
#elif defined(MATH21_FLAG_USE_CUDA)
    math21_function_activation_gradient_vector_cuda(y, n, a, dy);
#elif defined(MATH21_FLAG_USE_OPENCL)
    math21_function_activation_gradient_vector_opencl(y, n, a, dy);
#endif
}
