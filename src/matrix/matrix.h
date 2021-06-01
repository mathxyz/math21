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

#include "ten.h"

namespace math21 {
    /*Vector can be seen as matrix, but not vice versa.
     * So Mat may be a vector actually.
     * When Mat is not a vector, it can only access matrix methods.
     * When Mat is a vector, it can access both vector and matrix methods.
     * We can check it using dims(). That dims() is 1 means it's vector actually.
     * Vec is a vector, can be seen as matrix, so it can access both vector and matrix methods.
     *
     * */
}