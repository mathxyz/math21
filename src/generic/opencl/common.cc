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

#include "common.h"

#ifdef MATH21_FLAG_USE_OPENCL

using namespace math21;

_Map<std::string, std::string> mapFLikeName;

std::string math21_opencl_template_kernelName_using_suffix(const std::string &kernelName, const std::string &suffix) {
    return kernelName + "_" + suffix;
}

std::string math21_opencl_options_f_like(std::string d_function_ptr, std::string d_function_name) {
    if (mapFLikeName.isEmpty()) {
        mapFLikeName.add("f_shrink_min_like_ptr", "math21_device_f_min");
        mapFLikeName.add("f_shrink_argmin_like_ptr", "math21_device_f_argmin");
        mapFLikeName.add("f_bc_add_like_ptr", "math21_device_f_add");
        mapFLikeName.add("f_bc_sin_like_ptr", "math21_device_f_sin");
        mapFLikeName.add("f_kx_like_ptr", "math21_device_f_add");
        mapFLikeName.add("f_addto_like_ptr", "math21_device_f_addto");
        mapFLikeName.add("f_inner_product_like_ptr", "math21_device_f_inner_product");
        mapFLikeName.add("f_add_like_ptr", "math21_device_f_inner_product");
    } else {
        if (!d_function_ptr.empty()) {
            MATH21_ASSERT(mapFLikeName.has(d_function_ptr));
            mapFLikeName.valueAt(d_function_ptr) = d_function_name;
        }
    }
    std::string options = "";
    auto &dataMap = mapFLikeName.getData();
    for (auto itr = dataMap.begin(); itr != dataMap.end(); ++itr) {
        options += "-D " + itr->first + "=" + itr->second + " ";
    }
    return options;
}

#endif