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
#include "clprogram.h"
#include "cltool.h"

#ifdef MATH21_FLAG_USE_OPENCL

std::string math21_opencl_getPlatformInfoString(cl_platform_id platformId, cl_platform_info name);

std::shared_ptr<math21::m21clprogram>
math21_opencl_build_program_from_file(const std::string &sourcefileName, const std::string &options);

std::shared_ptr<math21::m21clprogram>
math21_opencl_build_program_from_two_files(const std::string &sourcefileName, const std::string &sourcefileName2,
                                           const std::string &options);

std::shared_ptr<math21::m21clprogram> math21_opencl_build_program_from_multiple_files(
        const std::vector<std::string> &sourcefileNames, const std::string &options);

cl_command_queue math21_opencl_get_command_queue();

cl_command_queue *math21_opencl_get_command_queue_pointer();

cl_context math21_opencl_get_context();

cl_kernel math21_opencl_getKernel(std::shared_ptr<math21::m21clprogram> &program, const std::string &kernelname);

std::shared_ptr<math21::m21clprogram>
math21_opencl_buildProgramFromFiles_detail(const std::vector<std::string> &sourcefileNames, const std::string &options,
                                           cl_device_id device, cl_context *context,
                                           std::map<std::string, std::string> &source_map);

std::shared_ptr<math21::m21cltool> math21_opencl_get_cltool();

void math21_opencl_vector_log_pointer(std::ostream &out, m21clvector v);

std::ostream &operator<<(std::ostream &out, const m21clvector &m);

#endif
