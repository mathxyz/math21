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

#include "var.h"

namespace math21 {

    void m21variable::clear() {
        _data_cpu = 0;
        _data_gpu = math21_vector_getEmpty_R32_wrapper();
        _data_wrapper = math21_vector_getEmpty_R32_wrapper();
        d.clear();
        A_cpu.clear();
    }

    void m21variable::update() {
        if (isEmpty()) {
            _size = 0;
        } else {
            _size = math21_operator_container_multiply_all(d);
        }
#if defined(MATH21_FLAG_USE_CPU)
        if(_data_cpu){
            _data_wrapper = _data_cpu;
        } else{
            _data_cpu = _data_wrapper;
        }
#else
        _data_gpu = _data_wrapper;
#endif
    }

    void m21variable::setName(const char *name) {
        _name = name;
    }

    const char *m21variable::getName() const {
        return _name.c_str();
    }

    m21variable::m21variable() {
        _data_cpu = 0;
        _data_gpu = math21_vector_getEmpty_R32_wrapper();
        _data_wrapper = math21_vector_getEmpty_R32_wrapper();
        _name = "";
    }

    m21variable::~m21variable() {

    }

    void m21variable::set(NumR32 *data, NumN d1) {
        clear();
        _data_cpu = data;
        d.setSize(1);
        d = d1;
        update();
    }

    void m21variable::set(NumR32 *data, NumN d1, NumN d2) {
        clear();
        _data_cpu = data;
        d.setSize(2);
        d = d1, d2;
        update();
    }

    void m21variable::set(NumR32 *data, NumN d1, NumN d2, NumN d3) {
        clear();
        _data_cpu = data;
        d.setSize(3);
        d = d1, d2, d3;
        update();
    }

    void m21variable::set(NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4) {
        clear();
        _data_cpu = data;
        d.setSize(4);
        d = d1, d2, d3, d4;
        update();
    }

    void m21variable::set(NumR32 *data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        clear();
        _data_cpu = data;
        d.setSize(5);
        d = d1, d2, d3, d4, d5;
        update();
    }

    void m21variable::setWrapper(PointerFloatWrapper data, NumN d1) {
        clear();
        _data_wrapper = data;
        d.setSize(1);
        d = d1;
        update();
    }

    void m21variable::setWrapper(PointerFloatWrapper data, NumN d1, NumN d2) {
        clear();
        _data_wrapper = data;
        d.setSize(2);
        d = d1, d2;
        update();
    }

    void m21variable::setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3) {
        clear();
        _data_wrapper = data;
        d.setSize(3);
        d = d1, d2, d3;
        update();
    }

    void m21variable::setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3, NumN d4) {
        clear();
        _data_wrapper = data;
        d.setSize(4);
        d = d1, d2, d3, d4;
        update();
    }

    void m21variable::setWrapper(PointerFloatWrapper data, NumN d1, NumN d2, NumN d3, NumN d4, NumN d5) {
        clear();
        _data_wrapper = data;
        d.setSize(5);
        d = d1, d2, d3, d4, d5;
        update();
    }

    NumB m21variable::isEmpty() const {
        return d.isEmpty();
    }

    void m21variable::log(const char *name) const {
        return log(std::cout, name);
    }

    void m21variable::log(std::ostream &io, const char *name) const {
        if (isEmpty()) {
            return;
        }
        if (_data_cpu) {
            math21_tensor_log_cpu(name, _data_cpu, d);
        } else {
            math21_tensor_log_wrapper(name, _data_gpu, d);
        }
    }

    NumB m21variable::getDataCpu(NumR32 *&p) const {
        if (isEmpty()) {
            return 0;
        }
        if (_data_cpu) {
            p = _data_cpu;
        } else {
            return 0;
        }
        return 1;
    }

    NumB m21variable::getDataWrapper(PointerFloatWrapper &p) const {
        if (isEmpty()) {
            return 0;
        }
        if (math21_vector_isEmpty_wrapper(_data_wrapper)) {
            return 0;
        } else {
            p = _data_wrapper;
        }
        return 1;
    }

    NumB m21variable::setDataCpu(const NumR32 *p) {
        if (isEmpty()) {
            return 0;
        }
        if (_data_cpu) {
            math21_vector_memcpy_cpu(_data_cpu, p, size() * sizeof(NumR32));
        } else {
            return 0;
        }
        return 1;
    }

    NumB m21variable::setDataWrapper(PointerFloatInputWrapper p) {
        if (isEmpty()) {
            return 0;
        }
        if (math21_vector_isEmpty_wrapper(_data_wrapper)) {
            return 0;
        } else {
            math21_vector_assign_from_vector_wrapper(size(), p, 1, _data_wrapper, 1);
        }
        return 1;
    }

    NumB m21variable::getDataToCpu(const NumR32 *&p) {
        if (isEmpty()) {
            return 0;
        }
        if (_data_cpu) {
            p = _data_cpu;
        } else {
            A_cpu.setSize(d);
            NumR32 *data_cpu = A_cpu.getDataAddress();
            math21_vector_pull_wrapper(_data_gpu, data_cpu, size());
            p = data_cpu;
        }
        return 1;
    }

    NumB m21variable::getRawTensorToCpu(m21rawtensor &p) {
        m21rawtensor rawtensor = {0};
        if (isEmpty()) {
            return 0;
        }
        rawtensor.type = m21_type_NumR32;
        rawtensor.dims = d.size();
        rawtensor.d = d.getDataAddress();
        if (_data_cpu) {
            rawtensor.data = _data_cpu;
        } else {
            A_cpu.setSize(d);
            NumR32 *data_cpu = A_cpu.getDataAddress();
            math21_vector_pull_wrapper(_data_gpu, data_cpu, size());
            rawtensor.data = data_cpu;
        }
        p = rawtensor;
        return 1;
    }

    NumB m21variable::setDataFromCpu(const NumR32 *p) {
        if (isEmpty()) {
            return 0;
        }
        if (_data_cpu) {
            math21_vector_memcpy_cpu(_data_cpu, p, size() * sizeof(NumR32));
        } else {
            math21_vector_push_wrapper(_data_gpu, p, size());
        }
        return 1;
    }

    NumSize m21variable::size() const {
        return _size;
    }

}
