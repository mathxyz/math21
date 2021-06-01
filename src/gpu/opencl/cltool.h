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
#include "inner_c.h"

#ifdef MATH21_FLAG_USE_OPENCL

#include<memory>
#include "clprogram.h"
#include "opencl_c.h"

namespace math21 {
    class m21cltool {
    private:
        cl_int error;
        static int instance_count;

        bool verbose;
        cl_context *context;
        cl_command_queue *queue;

        cl_platform_id platform_id;
        cl_device_id device;

        static void init_device(int gpuIndex, cl_platform_id &platform_id, cl_device_id &device);

        void init(int gpuIndex);

        void commonConstructor(cl_platform_id platform_id, cl_device_id device);

        template<typename T>
        static std::string toString(T val) {
            return math21::math21_string_to_string(val);
        }

        static std::string errorMessage(cl_int error);

        void finish();

    public:

        m21cltool(int gpu = 0);

        m21cltool(cl_platform_id platformId, cl_device_id deviceId);

        virtual ~m21cltool();

        cl_command_queue get_command_queue() const {
            math21_tool_assert(queue);
            return *queue;
        }

        cl_command_queue *get_command_queue_pointer() const {
            math21_tool_assert(queue);
            return queue;
        }

        cl_context get_context() const {
            math21_tool_assert(context);
            return *context;
        }

        void clear();

        void gpu(int gpuIndex);

        static int roundUp(int quantization, int minimum);

        static int getPower2Upperbound(int value);// eg pass in 320, it will return: 512

        //I would like to choose gpu,so ignore other device
        static std::shared_ptr<m21cltool> createForFirstGpu();

        static std::shared_ptr<m21cltool> createForIndexedGpu(int gpu);

        static m21cltool *createForIndexedGpu_raw(int gpu);

        static std::shared_ptr<m21cltool> createForPlatformDeviceIndexes(int platformIndex, int deviceIndex);

        static std::shared_ptr<m21cltool> createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId);

        // deprecated, use math21_opencl_build_program_from_file instead.
        std::shared_ptr<m21clprogram>
        buildProgramFromFiles(const std::vector<std::string> &sourcefileNames, const std::string &options);
    };
}

#endif
