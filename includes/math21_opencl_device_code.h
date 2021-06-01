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

// For nvidia drivers:
// kernel cache location: ~/.nv/ComputeCache in linux or %appdata%/NVIDIA/ComputeCache in windows
// Delete this directory after making a change to one of the include files, then it should force the driver to actually recompile the OpenCL kernel.
#include "../src/numbers/basic_math/level_02"