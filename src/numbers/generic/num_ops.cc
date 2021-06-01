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

#include "num_ops.h"

namespace math21 {
    NumB math21_check_clip(NumR &x) {
        NumB clipped = 0;
        if (x > MATH21_MAX) {
            clipped = 1;
        } else if (x < MATH21_MIN) {
            clipped = 1;
        }
        return clipped;
    }

    NumB math21_clip_not_less(NumN &x, const NumN &min) {
        NumB clipped = 0;
        if (x < min) {
            x = min;
            clipped = 1;
        }
        return clipped;
    }

    NumB math21_clip(NumN &x, const NumN &min, const NumN &max) {
        NumB clipped = 0;
        if (x > max) {
            x = max;
            clipped = 1;
        } else if (x < min) {
            x = min;
            clipped = 1;
        }
        return clipped;
    }

    NumB math21_clip(NumR &x, const NumR &min, const NumR &max) {
        NumB clipped = 0;
        if (x > max) {
            x = max;
            clipped = 1;
        } else if (x < min) {
            x = min;
            clipped = 1;
        }
        return clipped;
    }

    NumB math21_clip(NumR &x) {
        return math21_clip(x, MATH21_MIN, MATH21_MAX);
    }

}