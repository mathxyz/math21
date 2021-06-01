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

#include "inner.h"
#include "openmp_info.h"

namespace math21 {
    void math21_omp_info(std::ostream &io) {
#ifndef MATH21_FLAG_USE_OPENMP
        io << "OpenMP is unavailable!\n";
#else
        io << "the number of threads is " << omp_get_num_threads() << "\n";
        io << "the max number of threads is " << omp_get_max_threads() << "\n";
        io << "the number of processors is " << omp_get_num_procs() << "\n";
#endif
    }
}

