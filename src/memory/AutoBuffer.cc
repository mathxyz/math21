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

#include <iostream>
#include "../gpu/files.h"
#include "../matrix/vector/vector_wrapper.h"
#include "tool.h"
#include "AutoBuffer.h"

namespace math21 {
    SpaceParas::SpaceParas() {
        address = 0;
        start = 0;
        ref_count = 0;
        size = 0;
        unit = 0;
        type = m21_type_none;
        _deviceType = m21_device_type_default;
#if !defined(MATH21_FLAG_USE_CPU)
        address_wrapper = math21_vector_getEmpty_N8_wrapper();
        start_wrapper = math21_vector_getEmpty_N8_wrapper();
#endif
    }

    SpaceParas::~SpaceParas() {
    }

    void SpaceParas::log(const char *s) const {
        log(std::cout, s);
    }

    void SpaceParas::log(std::ostream &io, const char *s) const {
#if defined(MATH21_DISABLE_LOG)
        return;
#endif
        if (s) {
            io << s << " ";
        }
        io << "space paras:"
           << "\n\taddress: " << (void *) address
           << "\n\tstart: " << (void *) start
           << "\n\tref_count: " << (ref_count == 0 ? (0) : (*ref_count))
           << "\n\tsize: " << size
           << "\n\tunit: " << unit << " in byte (char)."
           << "\n\ttype: " << math21_type_name(type)
           << "\n\tis_cpu(): " << is_cpu();
#if !defined(MATH21_FLAG_USE_CPU)
        if (math21_vector_isEmpty_wrapper(address_wrapper)) {
            io << "\n\taddress_wrapper: " << "Empty"
               << "\n\tstart_wrapper: " << "Empty";
        } else {
            math21_tool_assert(!math21_vector_isEmpty_wrapper(start_wrapper));
            io << "\n\taddress_wrapper: " << address_wrapper
               << "\n\tstart_wrapper: " << start_wrapper;
        }

#endif
        io << std::endl;
    }

    NumB SpaceParas::isAddressEmpty() const {
        if (is_cpu()) {
            return !address ? 1 : 0;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            return math21_vector_isEmpty_wrapper(address_wrapper);
#else
            math21_tool_assert(0);
            return 0;
#endif
        }
    }

    NumB AutoBuffer::setSpace(const SpaceParas &paras) {
        if (paras.isAddressEmpty()) {
            return 0;
        }
        if (getDeviceType() != paras._deviceType) {
            clear();
        }
        setDeviceType(paras._deviceType);
        MATH21_ASSERT(paras.unit == sizeof(char),
                      "char has different size with space unit.");
        if (is_cpu()) {
            if (paras.address != space_address) {
                clear();
                space_address = paras.address;
                refcount = paras.ref_count;
                addref();
            }
            space_start = paras.start;
            space_size = paras.size;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            if (paras.address_wrapper != space_address_wrapper) {
                clear();
                space_address_wrapper = paras.address_wrapper;
                refcount = paras.ref_count;
                addref();
            }
            space_start_wrapper = paras.start_wrapper;
            space_size = paras.size;
#else
            math21_tool_assert(0);
#endif
        }
        type = paras.type;
        return 1;
    }

    // copy from cpu
    void AutoBuffer::dataCopyFromCpu(const void *data) {
        MATH21_ASSERT(!isEmpty() && data, "copy null data");
        if (is_cpu()) {
            math21_memory_memcpy(space_start, data, sizeof(char) * nn);
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            math21_vector_push_N8_wrapper(space_start_wrapper, (const NumN8 *) data, sizeof(char) * nn);
#else
            math21_tool_assert(0);
#endif
        }
    }

#if !defined(MATH21_FLAG_USE_CPU)

    // copy from gpu
    void AutoBuffer::dataCopyFromGpu(PointerN8InputWrapper data) {
        MATH21_ASSERT(!isEmpty() && !math21_vector_isEmpty_wrapper(data), "copy null data");
        if (is_cpu()) {
            math21_vector_pull_N8_wrapper(data, (NumN8 *) (space_start), sizeof(char) * nn);
        } else {
            math21_vector_assign_from_vector_N8_wrapper(sizeof(char) * nn, data, space_start_wrapper);
        }
    }

#endif

    SpaceParas AutoBuffer::getSpace() const {
        SpaceParas paras;
        paras.address = (char *) space_address;
        paras.start = (char *) space_start;
        paras.ref_count = refcount;
        paras.size = space_size;
        paras.unit = sizeof(char);
        paras.type = type;
        paras._deviceType = _deviceType;
#if !defined(MATH21_FLAG_USE_CPU)
        paras.address_wrapper = space_address_wrapper;
        paras.start_wrapper = space_start_wrapper;
#endif
        return paras;
    }

    void math21_memory_getSpace(const SpaceParas &src, SpaceParas &dst, NumN offset, NumN size, NumN unit) {
        MATH21_ASSERT(offset * unit + size * unit <= src.size)
        dst.ref_count = src.ref_count;
        dst.size = size * unit;
        dst.unit = src.unit;
        dst.type = src.type;
        dst._deviceType = src._deviceType;
        if (dst.is_cpu()) {
            dst.address = src.address;
            dst.start = src.start + offset * unit;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            dst.address_wrapper = src.address_wrapper;
            dst.start_wrapper = src.start_wrapper + offset * unit;
#else
            math21_tool_assert(0);
#endif
        }
    }

    SpaceParas AutoBuffer::getSpace(NumN offset, NumN size, NumN unit) const {
        MATH21_ASSERT(offset * unit + size * unit <= space_size)
        SpaceParas paras;
        if (is_cpu()) {
            paras.address = space_address;
            paras.start = space_start + offset * unit;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            paras.address_wrapper = space_address_wrapper;
            paras.start_wrapper = space_start_wrapper + offset * unit;
#else
            math21_tool_assert(0);
#endif
        }
        paras.ref_count = refcount;
        paras.size = size * unit;
        paras.unit = sizeof(char);
        paras.type = type;
        return paras;
    }

    void AutoBuffer::allocate() {
        if (is_cpu()) {
            if (space_address == 0) {
                math21_vector_malloc((void **) &space_address, sizeof(char) * nn);
//            xjmemset(space_address, 0, sizeof(char) * nn);

                MATH21_ASSERT((space_address != 0 && nn > 0) || (space_address == 0 && nn == 0));
                space_start = space_address;
                space_size = nn;
                if (space_address) {
                    math21_vector_malloc((void **) &refcount, sizeof(*refcount));
                    *refcount = 1;
                } else {
                    refcount = 0;
                }
            }
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            if (math21_vector_isEmpty_wrapper(space_address_wrapper)) {
                space_address_wrapper = (PointerN8Wrapper) math21_vector_create_buffer_wrapper(sizeof(char) * nn, 1);

                MATH21_ASSERT((!math21_vector_isEmpty_wrapper(space_address_wrapper) && nn > 0) ||
                              (math21_vector_isEmpty_wrapper(space_address_wrapper) && nn == 0));
                space_start_wrapper = space_address_wrapper;
                space_size = nn;
                if (!math21_vector_isEmpty_wrapper(space_address_wrapper)) {
                    math21_vector_malloc((void **) &refcount, sizeof(*refcount));
                    *refcount = 1;
                } else {
                    refcount = 0;
                }
            }
#else
            math21_tool_assert(0);
#endif
        }
    }

    void AutoBuffer::init() {
        space_address = 0;
        space_start = 0;
        space_size = 0;
        nn = 0;
        refcount = 0;
        name = "";
        type = m21_type_none;
        _deviceType = m21_device_type_default;
#if !defined(MATH21_FLAG_USE_CPU)
        space_address_wrapper = math21_vector_getEmpty_N8_wrapper();
        space_start_wrapper = math21_vector_getEmpty_N8_wrapper();
#endif
    }

    NumB AutoBuffer::isAddressEmpty() const {
        if (is_cpu()) {
            return !space_address ? 1 : 0;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            return math21_vector_isEmpty_wrapper(space_address_wrapper);
#else
            math21_tool_assert(0);
            return 0;
#endif
        }
    }

    NumB AutoBuffer::isStartEmpty() const {
        if (is_cpu()) {
            return !space_start ? 1 : 0;
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            return math21_vector_isEmpty_wrapper(space_start_wrapper);
#else
            math21_tool_assert(0);
            return 0;
#endif
        }
    }

    AutoBuffer::AutoBuffer() {
        init();
    }

    NumN AutoBuffer::size() const {
        return nn;
    }

    AutoBuffer::~AutoBuffer() {
        clear();
    }

    void AutoBuffer::deallocate() {
        math21_memory_free(refcount);
        if (is_cpu()) {
            math21_memory_free(space_address);
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            math21_vector_free_wrapper(space_address_wrapper);
#else
            math21_tool_assert(0);
#endif
        }
    }

    void AutoBuffer::setName(const char *_name) {
        name = _name;
    }

    void AutoBuffer::clear() {
        if (!isEmpty()) {
            if (refcount) {
//                m21log(name, *refcount);
                (*refcount)--;
                if (*refcount == 0) {
//                    m21log(name, "data deallocate");
                    deallocate();
                }
            }
            refcount = 0;
            nn = 0;
            type = m21_type_none;
            space_size = 0;
            if (is_cpu()) {
                space_address = 0;
                space_start = 0;
            } else {
#if !defined(MATH21_FLAG_USE_CPU)
                space_address_wrapper = math21_vector_getEmpty_N8_wrapper();
                space_start_wrapper = math21_vector_getEmpty_N8_wrapper();
#else
                math21_tool_assert(0);
#endif
            }
        }
    }

    NumB AutoBuffer::isIndependent() const {
        if (isEmpty() || (refcount && *refcount == 1)) {
            return 1;
        } else {
            return 0;
        }
    }

    // set size will set element zero.
    // Todo: remove xjmemset, don't clear to zero.
    void AutoBuffer::setSize(NumN n, NumN type, const SpaceParas *paras) {
        if (paras == 0) {
            MATH21_ASSERT(isIndependent(), "call ensureIndependence() or clear() first!\n"
                    << "\tname: " << name)
            if (n == 0 || isEmpty() || !isIndependent()) {
                clear();
                nn = n;
                allocate();
            } else {
                if (n != nn) {
                    if (is_cpu()) {
                        math21_memory_free(space_address);
                        math21_vector_malloc((void **) &space_address, sizeof(char) * n);
                        space_start = space_address;
                    } else {
#if !defined(MATH21_FLAG_USE_CPU)
                        space_address_wrapper = (PointerN8Wrapper) math21_vector_resize_buffer_wrapper(
                                (PointerVoidWrapper) space_address_wrapper, n, sizeof(char));
                        space_start_wrapper = space_address_wrapper;
#else
                        math21_tool_assert(0);
#endif
                    }
//                    xjrealloc((void**)&space_address, space_address, sizeof(char) * n);
//                    NumZ newStorage = n - nn;
//                    if (newStorage > 0)xjmemset(space_address + nn, 0, sizeof(char) * newStorage);
//                    xjmemset(space_address, 0, sizeof(char) * n);
                    nn = n;
                    space_size = nn;
                }
            }
            this->type = type;
        } else {
            setSpace(*paras);
            nn = n;
            MATH21_ASSERT(nn <= space_size, "space size mustn't be less than data size"
                    << "\n\trequired size: " << nn
                    << "\n\tgiven size: " << space_size);
        }
    }

    void AutoBuffer::addref() {
        if (refcount) {
            (*refcount)++;
        }
//        m21logDebug(*refcount);
    }

    void *AutoBuffer::getObjCpu() const {
        if (is_cpu()) {
            return (void *) space_start;
        } else {
            math21_tool_assert(0);
            return 0;
        }
    }

#if !defined(MATH21_FLAG_USE_CPU)

    PointerVoidWrapper AutoBuffer::getObjGpu() const {
        return space_start_wrapper;
    }

#endif

    NumN *AutoBuffer::getRefCount() const {
        return refcount;
    }

    int AutoBuffer::setDataShallow(const AutoBuffer &autoBuffer) {
        setSpace(autoBuffer.getSpace());
        nn = autoBuffer.size();
        return 1;
    }

    int AutoBuffer::setDataDeep(const AutoBuffer &autoBuffer) {
        if (size() != autoBuffer.size()) {
            setSize(autoBuffer.size(), autoBuffer.type);
        }
        if (autoBuffer.is_cpu()) {
            void *data;
            data = autoBuffer.getObjCpu();
            dataCopyFromCpu(data);
        } else {
#if !defined(MATH21_FLAG_USE_CPU)
            PointerN8Wrapper data;
            data = (PointerN8Wrapper) autoBuffer.getObjGpu();
            dataCopyFromGpu(data);
#else
            math21_tool_assert(0);
#endif
        }
        return 1;
    }

    void AutoBuffer::ensureIndependence() {
        if (!isEmpty() && !isIndependent()) {
            if (is_cpu()) {
                auto data = space_start;
                NumN _nn = size();
                clear();
                setSize(_nn, type);
                dataCopyFromCpu(data);
            } else {
#if !defined(MATH21_FLAG_USE_CPU)
                auto data = space_start_wrapper;
                NumN _nn = size();
                clear();
                setSize(_nn, type);
                dataCopyFromGpu(data);
#else
                math21_tool_assert(0);
#endif
            }
        }
    }

    NumB AutoBuffer::isEmpty() const {
        if (isAddressEmpty()) {
            MATH21_ASSERT(refcount == 0 && isStartEmpty(),
                          "error data is null but refcount isn't.");
            return 1;
        } else {
            MATH21_ASSERT((refcount == 0 || (refcount != 0 && *refcount > 0)),
                          "data is supposed to be not null, but"
                                  << "\n\trefcount is " << refcount
                                  << "\n\t*refcount is " << *refcount
                                  << "\n\tsize is " << size() << " (size is allowed not to be zero when is empty.)"
                                  << "\n\tgetClassName is " << getClassName());
            return 0;
        }
    }

    NumB AutoBuffer::is_cpu() const {
        if (_deviceType == m21_device_type_gpu) {
            return 0;
        } else {
            return 1;
        }
    }

    void AutoBuffer::setDeviceType(NumN deviceType) {
        MATH21_ASSERT(isEmpty(), "it must be called when empty.");
        _deviceType = deviceType;
    }

    NumN AutoBuffer::getDeviceType() const {
        return _deviceType;
    }

    std::string AutoBuffer::getClassName() {
        return "AutoBuffer";
    }

    void AutoBuffer::log(const char *s) const {
        log(std::cout, s);
    }

    void AutoBuffer::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = name.c_str();
        }
        io << "AutoBuffer " << s << ":\n";
        getSpace().log();
    }

    void AutoBuffer::swap(AutoBuffer &B) {
        m21_swap(space_address, B.space_address);
        m21_swap(space_start, B.space_start);
        m21_swap(space_size, B.space_size);
        m21_swap(nn, B.nn);
        m21_swap(refcount, B.refcount);
        m21_swap(type, B.type);
        m21_swap(_deviceType, B._deviceType);
#if !defined(MATH21_FLAG_USE_CPU)
        m21_swap(space_address_wrapper, B.space_address_wrapper);
        m21_swap(space_start_wrapper, B.space_start_wrapper);
#endif

    }

//    void AutoBuffer::zeros() {
//        if (!isEmpty()) {
//            xjmemset(space_start, 0, sizeof(char) * size());
//        }
//    }
}