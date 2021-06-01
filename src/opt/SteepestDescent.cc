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

#include "SteepestDescent.h"

using namespace math21;


SteepestDescent::SteepestDescent(sd_update_rule &update_rule, OptimizationInterface &oi) : update_rule(update_rule),
                                                                                           oi(oi) {
}

Functional &SteepestDescent::getFunctional() {
    return update_rule.f;
}

NumN SteepestDescent::getTime() {
    return update_rule.time;
}

void SteepestDescent::solve() {
    NumN stopTime = 0;
    while (1) {
        update_rule.update();
        if (xjabs(update_rule.y - update_rule.y_old) < XJ_EPS) {
            stopTime++;
            if (stopTime > 5) {
//                break;
            }
        } else {
            if (stopTime != 0) {
                stopTime = 0;
            }
        }
        if (update_rule.time % 1 == 0) {
            m21log("time", update_rule.time);
            m21log("y", update_rule.y_old);
        }
        update_rule.time++;
        oi.onFinishOneInteration(*this);
        if (update_rule.time >= update_rule.time_max) {
            break;
        }
    }
//    update_rule.x.log("minima");
}
