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

#include "../../probability/files_c.h"
#include "../../generic//files_c.h"
#include "update_wrapper.h"
#include "update.h"

OptAlphaPolicy math21_opt_learning_rate_policy_get_from_name(const char *s) {
    if (strcmp(s, "random") == 0) return OptAlphaPolicy_RANDOM;
    if (strcmp(s, "poly") == 0) return OptAlphaPolicy_POLY;
    if (strcmp(s, "constant") == 0) return OptAlphaPolicy_CONSTANT;
    if (strcmp(s, "step") == 0) return OptAlphaPolicy_STEP;
    if (strcmp(s, "exp") == 0) return OptAlphaPolicy_EXP;
    if (strcmp(s, "sigmoid") == 0) return OptAlphaPolicy_SIG;
    if (strcmp(s, "steps") == 0) return OptAlphaPolicy_STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return OptAlphaPolicy_CONSTANT;
}

float math21_opt_get_alpha_by_policy(m21OptAlphaPolicyConfig *config) {
    // t is time in opt.
    size_t t = config->t;
    int i;
    float alpha;
    // alpha <- alpha * x ^* 2, x in (0, 1), x = t/burn_in, so alpha is increased every time.
    if (t < config->burn_in) {
        alpha = (config->alpha * powf((float) t / config->burn_in, config->power));
        return alpha;
    }
    switch (config->alphaPolicy) {
        case OptAlphaPolicy_CONSTANT:
            return config->alpha;
        case OptAlphaPolicy_STEP:
            // alpha * scale ^* x, x = t / step
            return config->alpha * powf(config->scale, t / config->step);
        case OptAlphaPolicy_STEPS:
            // alpha is step function.
            alpha = config->alpha;
            for (i = 0; i < config->num_steps; ++i) {
                if (config->steps[i] > t) return alpha;
                alpha *= config->scales[i];
            }
            return alpha;
        case OptAlphaPolicy_EXP:
            // alpha * gamma ^* t
            return config->alpha * powf(config->gamma, t);
        case OptAlphaPolicy_POLY:
            // alpha * x ^* 2, x = 1-ratio, ratio = t/max_t
            return config->alpha * powf(1 - (float) t / config->n_mini_batch_max_in_opt, config->power);
        case OptAlphaPolicy_RANDOM:
            // alpha * x ^* 2, x is drawn from uni(0, 1)
            return config->alpha * powf(math21_pr_rand_uniform(0, 1), config->power);
        case OptAlphaPolicy_SIG:
            // alpha *(1/(1+ exp(gamma*x))), x = t - step
            return config->alpha * (1. / (1. + exp(config->gamma * (t - config->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return config->alpha;
    }
}

OptUpdate *math21_opt_update_create() {
    auto node = (OptUpdate *) math21_vector_calloc_cpu(1, sizeof(OptUpdate));
    return node;
}

void math21_opt_update_destroy(OptUpdate *node) {
    if (!node) {
        return;
    }
    math21_vector_free_cpu(node);
}

OptUpdate_Adam *math21_opt_update_adam_create() {
    auto node = (OptUpdate_Adam *) math21_vector_calloc_cpu(1, sizeof(OptUpdate_Adam));
    return node;
}

void math21_opt_update_adam_destroy(OptUpdate_Adam *node) {
    if (!node) {
        return;
    }
    math21_vector_free_cpu(node);
}

// deprecate, use math21_generic_optimization_adam_update_wrapper
void math21_optimization_adam_update_wrapper(PointerFloatWrapper x, PointerFloatWrapper neg_dx, PointerFloatWrapper m,
                                             PointerFloatWrapper v, float beta1, float beta2,
                                             float eps, float decay, float alpha, int x_size, int mini_batch_size,
                                             int t) {
    math21_generic_optimization_adam_update_wrapper(
            x, neg_dx, m, v, beta1, beta2, eps, decay, alpha, x_size, mini_batch_size, t, m21_type_NumR32);
#if 0
    // dL/dx = dL/dx + decay * x
    math21_vector_kx_add_y_wrapper(x_size, -decay * mini_batch_size, x, 1, neg_dx, 1);

    // update biased first moment estimate
    // m = beta1 * m + (1-beta1)*dL/dx
    // m = beta1 * m
    math21_vector_kx_wrapper(x_size, beta1, m, 1);
    // m = m + (1-beta1)*dL/dx
    math21_vector_kx_add_y_wrapper(x_size, (1 - beta1), neg_dx, 1, m, 1);

    // update biased second raw moment estimate
    // v = beta2 * v + (1-beta2)*(dL/dx * dL/dx)
    // v = beta2 * v
    math21_vector_kx_wrapper(x_size, beta2, v, 1);
    // dL/dx <- dL/dx * dL/dx
    math21_vector_xy_wrapper(x_size, neg_dx, 1, neg_dx, 1);
    // v = v + (1-beta2)*(dL/dx * dL/dx)
    math21_vector_kx_add_y_wrapper(x_size, (1 - beta2), neg_dx, 1, v, 1);

    math21_optimization_adam_update_part_2_wrapper(x_size, x, m, v, beta1, beta2, alpha, eps, t);
    // dL/dx <- 0
    math21_vector_set_wrapper(x_size, 0, neg_dx, 1);
#endif
}

// deprecated, use math21_op_optimization_adam_update instead.
// See [Kingma et al., 2014], https://arxiv.org/pdf/1412.6980.pdf
// adam update
// neg_dx = -dL/dx
// m is 1st moment vector, v is 2nd moment vector
// beta1, beta2, [0, 1): Exponential decay rates for the moment estimates
// alpha is learning rate, t is time step, eps is epsilon
void math21_generic_optimization_adam_update_wrapper(
        PointerVoidWrapper x, PointerVoidWrapper neg_dx, PointerVoidWrapper m,
        PointerVoidWrapper v, NumR beta1, NumR beta2,
        NumR eps, NumR decay, NumR alpha, NumN x_size, NumN mini_batch_size,
        NumN t, NumN type) {
    // dL/dx = dL/dx + decay * x
    if (decay != 0) {
        math21_generic_vector_kx_add_y_wrapper(x_size, -decay * mini_batch_size, x, 1, neg_dx, 1, type);
    }

    // update biased first moment estimate
    // m = beta1 * m + (1-beta1)*dL/dx
    // m = beta1 * m
    math21_generic_vector_kx_wrapper(x_size, beta1, m, 1, type);
    // m = m + (1-beta1)*dL/dx
    math21_generic_vector_kx_add_y_wrapper(x_size, (1 - beta1), neg_dx, 1, m, 1, type);

    // update biased second raw moment estimate
    // v = beta2 * v + (1-beta2)*(dL/dx * dL/dx)
    // v = beta2 * v
    math21_generic_vector_kx_wrapper(x_size, beta2, v, 1, type);
    // dL/dx <- dL/dx * dL/dx
    math21_generic_vector_xy_wrapper(x_size, neg_dx, 1, neg_dx, 1, type);
    // v = v + (1-beta2)*(dL/dx * dL/dx)
    math21_generic_vector_kx_add_y_wrapper(x_size, (1 - beta2), neg_dx, 1, v, 1, type);

    math21_generic_optimization_adam_update_part_2_wrapper(x_size, x, m, v, beta1, beta2, alpha, eps, t, type);
    // dL/dx <- 0
    math21_generic_vector_set_by_value_wrapper(x_size, 0, neg_dx, 1, type);
}