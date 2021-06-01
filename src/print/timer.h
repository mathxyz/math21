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

#include <thread>
#include <chrono>
#include "log.h"
#include "inner.h"

namespace math21 {
    struct timer {
    private:
        NumN count;
        NumB active;
#ifdef MATH21_FLAG_USE_CUDA
        void *_start;
        void *_stop;
        float elapsed_time;
#else
        NumR t;
#endif

        void stop();

    public:

        timer();

        virtual ~timer();

        void start();

        void end();

        // ms
        NumR time() const;

        template<typename T>
        void setTimeout(T function, int delay);

        template<typename T>
        void setInterval(T function, int interval);

        void log(const char *name = 0) const;
    };

    template<typename T>
    void timer::setTimeout(T function, int delay) {
#ifdef MATH21_FLAG_USE_THREAD
        this->active = 1;
        std::thread t([=]() {
            if (!this->active) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            if (!this->active) return;
            function();
            stop();
        });
        t.detach();
#else
        m21warn("Thread not disabled!");
#endif
    }

    template<typename T>
    void timer::setInterval(T function, int interval) {
#ifdef MATH21_FLAG_USE_THREAD
        this->active= 1;
        std::thread t([=]() {
            while (true) {
                if (!this->active) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(interval));
                if (!this->active) return;
                function();
                ++count;
                printf("timer: %d\n", count);

            }
        });
        t.detach();
#else
        m21warn("Thread not disabled!");
#endif
    }

#if defined(MATH21_ENABLE_PRINT_TIME)

#define MATH21_PRINT_TIME_ELAPSED(X) \
            timer theTimer;\
            theTimer.start();\
            X;\
            theTimer.end();\
            if (theTimer.time() > 0) {m21log(math21_string_to_string("time elapsed of {", MATH21_STRINGIFY(X), "}"), theTimer.time());}
#else

#define MATH21_PRINT_TIME_ELAPSED(X) X;

#endif
}