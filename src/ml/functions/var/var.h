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

#include "inner_cc.h"

namespace math21 {
    struct m21variable {
    private:
        std::string _name;
        PointerFloatWrapper _data_wrapper;
        PointerFloatGpu _data_gpu;
        NumR32 *_data_cpu;
        VecN d; // shape
        Tensor <NumR32> A_cpu;
        NumSize _size;
    public:
        void clear();

        void update();

        void setName(const char* name);

        const char* getName()const;

        m21variable();

        virtual ~m21variable();

        void set(NumR32 *data, NumN d1);

        void set(NumR32 *data, NumN d1, NumN d2);

        void set(NumR32 *data, NumN d1, NumN d2, NumN d3);

        void set(NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4);

        void set(NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5);

        void setWrapper(PointerFloatWrapper data, NumN d1);

        void setWrapper(PointerFloatWrapper data, NumN d1, NumN d2);

        void setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3);

        void setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3, NumN d4);

        void setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5);

        NumB isEmpty() const;

        void log(const char *name = 0) const;

        void log(std::ostream &io, const char *name = 0) const;

        NumB getDataCpu(NumR32 *&p) const;

        NumB getDataWrapper(PointerFloatWrapper &p) const;

        NumB setDataCpu(const NumR32 *p);

        NumB setDataWrapper(PointerFloatInputWrapper p);

        NumB getDataToCpu(const NumR32 *&p);

        NumB getRawTensorToCpu(m21rawtensor &p);

        NumB setDataFromCpu(const NumR32 *p);

        NumSize size() const;
    };

}
