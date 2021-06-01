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

#include "clprogram.h"

#ifdef MATH21_FLAG_USE_OPENCL

namespace math21 {

    m21clprogram::m21clprogram(cl_program prog) : program(prog) {
//        m21log(__FUNCTION__);
    }

    cl_kernel m21clprogram::getKernel(const std::string &kernelname) {
        auto iter = kernels.find(kernelname);
        if (iter == kernels.end()) {
            cl_int error;
            cl_kernel kernel = clCreateKernel(program, kernelname.c_str(), &error);
            if (error != CL_SUCCESS) {
                std::string exceptionMessage = "";
                switch (error) {
                    case -46:
                        exceptionMessage = "Invalid kernel name, code -46, kernel " + kernelname + "\n";
                        break;
                    default:
                        exceptionMessage = "Something went wrong with clCreateKernel, OpenCL error code " +
                                           math21_string_to_string(error) + "\n";
                        break;
                }

                std::cout << "kernel build error:\n" << exceptionMessage << std::endl;
                MATH21_ASSERT(0, exceptionMessage);
            }
            kernels[kernelname] = kernel;
            return kernel;
        } else {
            return iter->second;
        }
    }

    m21clprogram::~m21clprogram() {
        clear();
    }

    void m21clprogram::clear() {
//        m21log("m21clprogram clear");
        if (isEmpty()) {
            return;
        }
        for (auto iter = kernels.begin(); iter != kernels.end(); iter++) {
            clReleaseKernel(iter->second);
        }
        kernels.clear();
        if (program) {
            clReleaseProgram(program);
            program = 0;
        }
    }

    NumB m21clprogram::isEmpty() const {
        if (program) {
            return 0;
        }
        return 1;
    }
}

#endif