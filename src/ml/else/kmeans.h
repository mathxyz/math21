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

#include "inner.h"


namespace math21 {

    struct ml_kmeans_config {
    public:
        NumN total_points, total_values, K, max_iterations;

        ml_kmeans_config(NumN K, NumN total_points, NumN total_values, NumN max_iterations) {
            this->K = K;
            this->total_points = total_points;
            this->total_values = total_values;
            this->max_iterations = max_iterations;
        }
    };


    void ml_kmeans(const Seqce <TenR> &data, VecN &labels,
                   const ml_kmeans_config &config);

    void ml_kmeans(const Seqce <TenR> &data, VecN &labels, VecN &num_in_clusters,
                   const ml_kmeans_config &config);

    void ml_kmeans(const Seqce <TenR> &data, VecN &labels, Seqce <VecN> &points_in_clusters,
                   const ml_kmeans_config &config);
}