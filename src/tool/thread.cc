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

#include <thread>
//#include <pthread.h>
#include <mutex>
#include "thread_c.h"

static std::mutex m21mutex;
//pthread_mutex_t m21mutex = PTHREAD_MUTEX_INITIALIZER;

void *math21_tool_thread_create() {
    auto *t = new std::thread();
//    pthread_t *t = (pthread_t * )(math21_vector_calloc_cpu(1, sizeof(pthread_t)));
    return t;
}

void math21_tool_thread_start(void *t, const void *attr, void *(*start_routine)(void *), void *arg) {
    *((std::thread *) t) = std::thread(start_routine, arg);
//    if (pthread_create((pthread_t *) t, attr, start_routine, arg)) math21_error("Thread creation failed");
}

void math21_tool_thread_just_join(void *_t, void **r) {
    std::thread &t = *(std::thread *) (_t);
    t.join();
//    pthread_t &t = *(pthread_t * )(_t);
//    pthread_join(t, r);
}

void math21_tool_thread_join_and_destroy(void *_t, void **r) {
    std::thread &t = *(std::thread *) (_t);
    t.join();
    delete (std::thread *) (_t);
//    pthread_t &t = *(pthread_t * )(_t);
//    pthread_join(t, r);
//    free((pthread_t * )(_t));
}

void math21_tool_thread_destroy(void *_t) {
    delete (std::thread *) (_t);
//    free((pthread_t * )(_t));
}

void math21_tool_thread_mutex_lock() {
    m21mutex.lock();
//    pthread_mutex_lock(&m21mutex);
}

void math21_tool_thread_mutex_unlock() {
    m21mutex.unlock();
//    pthread_mutex_unlock(&m21mutex);
}
