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

#include "num_add.h"
#include "num_multiply.h"
#include "num_assign.h"
#include "num_constant.h"
#include "num_input.h"

#include "vec_add.h"
#include "vec_multiply.h"

#include "mat_eye.h"
#include "mat_block_diag_same.h"
#include "op_mat_mul.h"
#include "op_mat_trans.h"
#include "mat_jacobian.h"

#include "op_do_nothing.h"
#include "op_multiply.h"

#include "op_sin.h"
#include "op_sum.h"
#include "op_multiply.h"
#include "op_add.h"
#include "op_assign.h"

#include "op_share_reshape.h"
#include "op_get_shape.h"

#include "ad_global.h"