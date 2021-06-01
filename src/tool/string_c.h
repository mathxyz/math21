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

#include "inner_c.h"

#ifdef __cplusplus
extern "C" {
#endif

NumN math21_string_length(const char *s);

NumB math21_string_get_file_name(const char *s, char *buffer, NumN bufferSize);

NumB math21_string_get_file_name_without_suffix(const char *s, char *buffer, NumN bufferSize);

void math21_string_strip(char *s);

unsigned char *math21_string_read_file(const char *filename);

void math21_string_c_free(unsigned char *text);

const char *math21_string_create_from_string(const char *s);

#ifdef __cplusplus
}
#endif
