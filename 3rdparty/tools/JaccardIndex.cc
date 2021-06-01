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

#include <math.h>
#include <cstdio>
#include "JaccardIndex.h"

// rect 1: x11, x12, y11, y12; rect 2: x21, x22, y21, y22
float JaccardIndex(float x11, float x12, float y11, float y12,
                   float x21, float x22, float y21, float y22
) {
    double ai, a1, a2;
    double xi1, xi2, yi1, yi2;
    xi1 = fmax(x11, x21);
    xi2 = fmin(x12, x22);
    yi1 = fmax(y11, y21);
    yi2 = fmin(y12, y22);
    if(xi1>=xi2 || yi1>=yi2){
        return 0;
    }
    ai = fabs(xi2 - xi1) * fabs(yi2 - yi1);
    a1 = fabs(x12 - x11) * fabs(y12 - y11);
    a2 = fabs(x22 - x21) * fabs(y22 - y21);
    if (a1 + a2 - ai == 0) {
        return 1;
    }
    return (float) (ai / (a1 + a2 - ai));
}
