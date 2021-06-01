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

/*
 * AutoBuffer is used for basic types only (as opposed to non-basic types).
 * It creates a continuous memory using malloc or calloc.
 * See BufferNew
 * */
namespace math21 {

    // here char* is actually void*, because it may be used for non-basic types.
    // the unit is meaningless for non-basic types.
    struct SpaceParas {
        char *address;//new and delete address.
        char *start;//available space start position.
        NumN *ref_count;
        NumN size;//size space is available.
        NumN unit;//unit size in byte, char
        NumN type;
        NumN _deviceType;
#if !defined(MATH21_FLAG_USE_CPU)
        PointerN8Wrapper address_wrapper;
        PointerN8Wrapper start_wrapper;
#endif

        SpaceParas();

        virtual ~SpaceParas();

        void log(const char *s = 0) const;

        void log(std::ostream &io, const char *s = 0) const;

        NumB isAddressEmpty() const;

        NumB is_cpu() const {
            if (_deviceType == m21_device_type_gpu) {
                return 0;
            } else {
                return 1;
            }
        }
    };

    void math21_memory_getSpace(const SpaceParas &src, SpaceParas &dst, NumN offset, NumN size, NumN unit);

    /*
     * 1. setSize: allocate or set space.
     * 2.
     * */
    // space_unit is char.
    struct AutoBuffer {
    private:
//        SpaceParas paras;
        char *space_address; // address of all allocated buffer.
        char *space_start; // start address given to this buffer.
        NumN space_size; // space_size is the size given to this buffer, not all allocated buffer size.
        NumN nn;// size used by this buffer, nn <= space_size, but the two values often equal to each other.
        NumN *refcount;// when matrix points to user-allocated data, the pointer is NULL
        NumN type;
        NumN _deviceType;
#if !defined(MATH21_FLAG_USE_CPU)
        PointerN8Wrapper space_address_wrapper;
        PointerN8Wrapper space_start_wrapper;
#endif

        /////////////
        std::string name;

        void init();

        NumB isAddressEmpty() const;

        NumB isStartEmpty() const;

        void deallocate();

        //Todo: clean
        void allocate();

        void addref();

        NumB setSpace(const SpaceParas &paras);

        void dataCopyFromCpu(const void *data);

#if !defined(MATH21_FLAG_USE_CPU)

        void dataCopyFromGpu(PointerN8InputWrapper data);

#endif

    public:

        AutoBuffer();

        virtual ~AutoBuffer();

        virtual SpaceParas getSpace() const;

        // unit is input unit, but char is used as unit inside the buffer.
        virtual SpaceParas getSpace(NumN offset, NumN n, NumN unit = sizeof(char)) const;

        //Construct n*m matrix with element set to zero using xjcalloc. Fast.
        //must set to zero. Other functions rely on this.
        // data is not kept.
        void setSize(NumN n, NumN type, const SpaceParas *paras = 0);

        //Todo: copy all headers including start position, end position...
        int setDataShallow(const AutoBuffer &autoBuffer);

        int setDataDeep(const AutoBuffer &autoBuffer);

        /*
         * The step which call setDataDeep will be efficient than
         * the steps which call ensureIndependence, then call setDataDeep.
         * So, don't use this if necessary.
         * */
        void ensureIndependence();

        void *getObjCpu() const;

#if !defined(MATH21_FLAG_USE_CPU)

        PointerVoidWrapper getObjGpu() const;

#endif

        NumN *getRefCount() const;

        NumN size() const;

        void clear();

        //return false if in share mode, or data is set by unknowns.
        NumB isIndependent() const;

        //only for basic types.
//        void zeros();

        void setName(const char *_name);

        void log(const char *name = 0) const;

        void log(std::ostream &io, const char *name = 0) const;

        // check if data is null. m or n may not be 0 if data is null. So use this function to check instead of m and n.
        NumB isEmpty() const;

        NumB is_cpu() const;

        void setDeviceType(NumN deviceType);

        NumN getDeviceType() const;

        static std::string getClassName();

        void swap(AutoBuffer &B);
    };
}