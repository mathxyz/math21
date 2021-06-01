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

#include "../memory/files.h"
#include "tool.h"
#include "timer.h"

namespace math21 {
    timer::timer() {
//        printf("timer create\n");
        count = 0;
        active = 1;
//        log("my");
#ifdef MATH21_FLAG_USE_CUDA
        _start = new cudaEvent_t();
        _stop = new cudaEvent_t();
#endif
    }

    timer::~timer() {
#ifdef MATH21_FLAG_USE_CUDA
        delete (cudaEvent_t *) _start;
        delete (cudaEvent_t *) _stop;
#endif
        count = 0;
        active = 0;
    }

    void timer::start() {
#ifdef MATH21_FLAG_USE_CUDA
        math21_cuda_EventCreate(_start);
        math21_cuda_EventCreate(_stop);
        cudaStream_t stream = 0;
        math21_cuda_EventRecord(_start, &stream);
#else
        t = math21_time_getticks();
#endif
    }

    void timer::end() {
#ifdef MATH21_FLAG_USE_CUDA
        cudaStream_t stream = 0;
        math21_cuda_EventRecord(_stop, &stream);
        math21_cuda_EventSynchronize(_stop);
        math21_cuda_EventElapsedTime(&elapsed_time, _start, _stop);
#else
        t = math21_time_getticks() - t;
#endif
    }

    NumR timer::time() const {
#ifdef MATH21_FLAG_USE_CUDA
        return (NumR) elapsed_time;
#else
        return t * 1000;
#endif
    }

    void timer::stop() {
        this->active = 0;
    }

    void timer::log(const char *name) const {
        if (name) {
            printf("%s\n", name);
        }
        printf("is active =%d\n", active);
        printf("count=%d\n", count);
    }
}