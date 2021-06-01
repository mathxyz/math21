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

#ifdef MATH21_ANDROID

#include <jni.h>
#include <android/log.h>

#endif

#include <iostream>
#include <streambuf>
#include <cstdio>

namespace math21 {
    class SimpleStreamBuf : public std::streambuf {
        enum {
            BUFFER_SIZE = 255,
        };

    public:
        SimpleStreamBuf() {
            buffer_[BUFFER_SIZE] = '\0';
            setp(buffer_, buffer_ + BUFFER_SIZE - 1);
        }

        ~SimpleStreamBuf() {
            sync();
        }

    protected:
        virtual int_type overflow(int_type c) {
            if (c != EOF) {
                *pptr() = static_cast<char_type >(c);
                pbump(1);
            }
            flush_buffer();
            return c;
        }

        virtual int sync() {
            flush_buffer();
            return 0;
        }

    private:
        int flush_buffer() {
            int len = int(pptr() - pbase());
            if (len <= 0)
                return 0;

            if (len <= BUFFER_SIZE) {
                buffer_[len] = '\0';
            }

#ifdef MATH21_ANDROID
            const char *TAG = "math21";
            android_LogPriority t = ANDROID_LOG_INFO;
            __android_log_write(t, TAG, buffer_);
#else
            printf("%s", buffer_);
#endif

            pbump(-len);
            return len;
        }

    private:
        char buffer_[BUFFER_SIZE + 1];
    };
}