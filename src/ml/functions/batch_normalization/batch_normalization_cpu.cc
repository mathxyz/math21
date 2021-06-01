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

#include "batch_normalization_cpu.h"
#ifdef MATH21_FLAG_USE_CPU
// error: dL/dmu = sum(dL/dX_hat(i) * dX_hat(i)/dmu)
// Todo: dL/dmu = sum(dL/dX_hat(i) * dX_hat(i)/dmu) + dL/dsigma_square * dsigma_square/dmu
void math21_ml_batchnormalization_backward_mu_cpu(const float *dX_hat, const float *variance, int mini_batch_size, int features_size, int in_class_size, float *dmu)
{

    int imb,ifeature,imember;
    for(ifeature = 0; ifeature < features_size; ++ifeature){
        dmu[ifeature] = 0;
        for (imb = 0; imb < mini_batch_size; ++imb) {
            for (imember = 0; imember < in_class_size; ++imember) {
                int index = (imb*features_size + ifeature)*in_class_size + imember;
                dmu[ifeature] += dX_hat[index];
            }
        }
        dmu[ifeature] *= (-1.f/sqrtf(variance[ifeature] + .00001f));
    }
}

// dL/dsigma_square = sum(dL/dX_hat(i) * dX_hat(i)/dsigma_square)
void math21_ml_batchnormalization_backward_sigma_square_cpu(const float *X, const float *dX_hat, const float *mu, const float *variance,
                                                            int mini_batch_size, int features_size, int in_class_size, float *dvariance)
{
    int imb,ifeature, imember;
    for(ifeature = 0; ifeature < features_size; ++ifeature){
        dvariance[ifeature] = 0;
        for(imb = 0; imb < mini_batch_size; ++imb){
            for(imember = 0; imember < in_class_size; ++imember){
                int index = (imb*features_size + ifeature)*in_class_size + imember;
                dvariance[ifeature] += dX_hat[index]*(X[index] - mu[ifeature]);
            }
        }
        dvariance[ifeature] *= -.5 * powf(variance[ifeature] + .00001f, (float)(-3./2.));
    }
}

// dL/dX(i) = dL/dX_hat(i) * dX_hat(i)/dX(i) + dL/dsig_square * dsig_square/dX(i) + dL/dmu * dmu/dX(i)
void math21_ml_batchnormalization_backward_input_cpu(const float *X, const float *mu, const float *variance, const float *dmu, const float *dvariance, int mini_batch_size, int features_size, int in_class_size, float *dX_hat)
{
    int imb,ifeature, imember;
    for(imb = 0; imb < mini_batch_size; ++imb){
        for(ifeature = 0; ifeature < features_size; ++ifeature){
            for(imember = 0; imember < in_class_size; ++imember){
                int index = (imb*features_size + ifeature)*in_class_size + imember;
                dX_hat[index] = dX_hat[index] * 1./(sqrt(variance[ifeature] + .00001f)) + dvariance[ifeature] * 2. * (X[index] - mu[ifeature]) / (mini_batch_size * in_class_size) + dmu[ifeature]/(mini_batch_size*in_class_size);
            }
        }
    }
}
#endif