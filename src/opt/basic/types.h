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

#pragma once

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float beta1;
    float beta2;
    float eps;
    int t;
} OptUpdate_Adam;

typedef enum {
    OptUpdateType_None, OptUpdateType_Adam, OptUpdateType_AdaGrad, OptUpdateType_RMSProp, OptUpdateType_LevMar
} OptUpdateType;

typedef struct {
    OptUpdateType type;
    float alpha; // learning_rate
    float momentum;
    float decay;
    int mini_batch_size;
    void *detail;
} OptUpdate;

typedef enum {
    OptAlphaPolicy_CONSTANT, OptAlphaPolicy_STEP, OptAlphaPolicy_EXP, OptAlphaPolicy_POLY,
    OptAlphaPolicy_STEPS, OptAlphaPolicy_SIG, OptAlphaPolicy_RANDOM
} OptAlphaPolicy;

struct m21OptAlphaPolicyConfig{
    int n_mini_batch_max_in_opt;
    size_t t;
    OptAlphaPolicy alphaPolicy;
    int burn_in; // alpha
    float alpha; // learning rate
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;

    // OptAlphaPolicy_STEPS
    int num_steps;
    int *steps;
    float *scales;

    float gamma;
    int step;
    float power;
    float scale;
};

typedef struct m21OptAlphaPolicyConfig m21OptAlphaPolicyConfig;

#ifdef __cplusplus
}
#endif
