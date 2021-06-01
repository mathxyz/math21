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

#include <map>
#include "inner.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {
    class m21clprogram {
    public:
        explicit m21clprogram(cl_program);

        cl_kernel getKernel(const std::string &kernelname);

        ~m21clprogram();

        void clear();

        NumB isEmpty() const;

    private:
        std::map<std::string, cl_kernel> kernels;
        cl_program program;
    };

}
#endif