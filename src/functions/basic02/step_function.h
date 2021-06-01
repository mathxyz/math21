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

#include "inner.h"

namespace math21 {
    struct Step {
    private:
        void init() {
            values.clear();
            intervals.clear();
            _isStandard = 0;
        }

        NumB _isStandard;

        void toStandard_no_overlap();

        void remove_overlap();

    public:
        VecR values;
        Seqce<Interval> intervals;

        const VecR &getValues() const {
            return values;
        }

        const Seqce<Interval> &getIntervals() const {
            return intervals;
        }
// intervals from left to right.

        void clear() {
            MATH21_ASSERT(values.size() == intervals.size())
            _isStandard = 0;
            values.clear();
            intervals.clear();
        }

        void add(const Step &g, Step &h) const;

        Step() {
            init();
            setZero();
        }

        virtual ~Step() {
            clear();
        }

        void setSize(NumN n) {
            clear();
            values.setSize(n);
            intervals.setSize(n);
        }

        // return number of intervals.
        NumN size() const {
            return values.size();
        }

        NumB isZero() const {
            if (size() == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        void setZero() {
            clear();
            setStandard(1);
        }

        NumR valueAt(NumR x) const {
            if (isZero()) {
                return 0;
            }
            NumR y = 0;
            for (NumN i = 1; i <= getIntervals().size(); ++i) {
                const Interval &I = getIntervals()(i);
                if (I.isInclude(x)) {
                    y = getValues()(i);
                    return y;
                }
            }
            return y;
        }

        // heavy, use less
        void push(NumR value, const Interval &interval) {
            VecR tmp;
            math21_operator_vec_insert_value(values, tmp, values.size() + 1, value);
            values.swap(tmp);
            intervals.push(interval);
        }

        void set(NumN i, NumR value, const Interval &interval) {
            MATH21_ASSERT(i <= size())
            values(i) = value;
            intervals(i) = interval;
        }

        void log(const char *s = 0) const {
            if (s) {
                std::cout << "step function " << s << ":\n";
                std::cout << "there are " << size() << " intervals.\n";
                std::cout << "Is standard: " << isStandard() << "\n";
            }
            for (NumN i = 1; i <= size(); ++i) {
                std::cout << values.operator()(i) << ", in ";
                intervals.operator()(i).log();
            }
        }

        void copyTo(Step &step) const {
            if (isZero()) {
                step.setZero();
                return;
            }
            step.clear();
            values.copyTo(step.values);
            intervals.copyTo(step.intervals);
            step.setStandard(isStandard());
        }

        void swap(Step &step) {
            values.swap(step.values);
            intervals.swap(step.intervals);
            m21_swap(_isStandard, step._isStandard);
        }

        NumB isStandard() const {
            return _isStandard;
        }

        void setStandard(NumB a) {
            _isStandard = a;
        }

        void toStandard();
    };

// h = f + g
    void math21_function_step_add(const Step &f, const Step &g, Step &h);

    void test_step_function();
}