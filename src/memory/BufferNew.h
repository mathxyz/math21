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

#include <iostream>
#include "tool.h"
#include "AutoBuffer.h"
#include "inner.h"

/*
 * BufferNew is used for non-basic types only (as opposed to basic types).
 * See AutoBuffer
 * Although BufferNew can be used for all types, but it is used for non-basic types currently.
 * And AutoBuffer is used for basic types.
 * */
namespace math21 {

    template<typename T>
    struct BufferNew {
    private:
        // The unit of size is meaningless. It just means the number of objects.
        T *space_address; // address of all allocated buffer.
        T *space_start; // start address given to this buffer.
        NumN space_size; // space_size is the size given to this buffer, not all allocated buffer size.
        NumN nn;// size used by this buffer, nn <= space_size, but the two values often equal to each other.
        NumN *refcount;// when matrix points to user-allocated data, the pointer is NULL

        /////////////
        std::string name;

        void init();

        void deallocate();

        //Todo: clean
        void allocate();

        void addref();

        NumB setSpace(const SpaceParas &paras);

        void dataCopy(const void *data);

    public:

        BufferNew();

        virtual ~BufferNew();

        virtual SpaceParas getSpace() const;

        // unit is input unit, but char is used as unit inside the buffer.
        virtual SpaceParas getSpace(NumN offset, NumN n, NumN unit = sizeof(char)) const;

        int setDataDeep(const BufferNew &autoBuffer);

        /*
         * The step which call setDataDeep will be efficient than
         * the steps which call ensureIndependence, then call setDataDeep.
         * So, don't use this if necessary.
         * */
        void ensureIndependence();

        //Todo: copy all headers including start position, end position...
        int setDataShallow(const BufferNew &autoBuffer);

        void *getObj() const;

        NumN *getRefCount() const;

        NumN size() const;

        void clear();

        //return false if in share mode, or data is set by unknowns.
        NumB isIndependent() const;

        //Construct n*m matrix with element set to zero using xjcalloc. Fast.
        //must set to zero. Other functions rely on this.
        // data is not kept.
        void setSize(NumN n, const SpaceParas *paras = 0);

        //only for basic types.
//        void zeros();

        void setName(const char *_name);

        void log(const char *name = 0) const;

        void log(std::ostream &io, const char *name = 0) const;

        // check if data is null. m or n may not be 0 if data is null. So use this function to check instead of m and n.
        bool isEmpty() const;

        std::string getClassName() const;

        void swap(BufferNew &B);
    };

    // space unit is meaningless.
    template<typename T>
    NumB BufferNew<T>::setSpace(const SpaceParas &paras) {
        if (paras.address == 0) {
            return 0;
        }
        MATH21_ASSERT(paras.unit == 1,
                      "Space unit is meaningless, so it is specified to be 1.");
        if ((T *) paras.address != space_address) {
            clear();
            space_address = (T *) paras.address;
            refcount = paras.ref_count;
            addref();
        }
        space_start = (T *) paras.start;
        space_size = paras.size;
        return 1;
    }

    // see std::copy
    // assignment in object
    template<typename T>
    void BufferNew<T>::dataCopy(const void *data0) {
        MATH21_ASSERT(!isEmpty() && data0, "copy null data");
        const T *data = (const T *) data0;
        NumN i;
        for (i = 0; i < nn; ++i) {
            space_start[i] = data[i];
        }
    }

    template<typename T>
    SpaceParas BufferNew<T>::getSpace() const {
        SpaceParas paras;
        paras.address = (char *) space_address;
        paras.start = (char *) space_start;
        paras.ref_count = refcount;
        paras.size = space_size;
        paras.unit = sizeof(char);
        return paras;
    }

    template<typename T>
    SpaceParas BufferNew<T>::getSpace(NumN offset, NumN size, NumN unit) const {
        MATH21_ASSERT(offset * unit + size * unit <= space_size)
        SpaceParas paras;
        paras.address = (char *) space_address;
        paras.start = (char *) space_start + offset * unit;
        paras.ref_count = refcount;
        paras.size = size * unit;
        paras.unit = sizeof(char);
        return paras;
    }

    template<typename T>
    void BufferNew<T>::allocate() {
        if (space_address == 0) {
            space_address = new T[nn];
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
    }


    template<typename T>
    void BufferNew<T>::init() {
        space_address = 0;
        space_start = 0;
        space_size = 0;
        nn = 0;
        refcount = 0;
        name = "";
    }

    template<typename T>
    BufferNew<T>::BufferNew() {
        init();
    }

    template<typename T>
    NumN BufferNew<T>::size() const {
        return nn;
    }

    template<typename T>
    BufferNew<T>::~BufferNew() {
        clear();
    }

    template<typename T>
    void BufferNew<T>::deallocate() {
        if (space_address) {
            delete[] space_address;
            space_address = 0;
        }
        math21_memory_free(refcount);
    }


    template<typename T>
    void BufferNew<T>::setName(const char *_name) {
        name = _name;
    }

    template<typename T>
    void BufferNew<T>::clear() {
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
            space_address = 0;
            space_start = 0;
            space_size = 0;
        }
    }

    template<typename T>
    NumB BufferNew<T>::isIndependent() const {
        if (isEmpty() || (refcount && *refcount == 1)) {
            return 1;
        } else {
            return 0;
        }
    }

    template<typename T>
    void BufferNew<T>::setSize(NumN n, const SpaceParas *paras) {
        if (paras == 0) {
            MATH21_ASSERT(isIndependent(), "call ensureIndependence() or clear() first!\n"
                    << "\tname: " << name)
            if (n == 0 || isEmpty() || !isIndependent()) {
                clear();
                nn = n;
                allocate();
            } else {
                if (n != nn) {
                    space_address = new T[n];
                    nn = n;
                    space_start = space_address;
                    space_size = nn;
                }
            }
        } else {
            setSpace(*paras);
            nn = n;
            MATH21_ASSERT(nn <= space_size, "space size mustn't be less than data size"
                    << "\n\trequired size: " << nn
                    << "\n\tgiven size: " << space_size);
        }
    }

    template<typename T>
    void BufferNew<T>::addref() {
        if (refcount) {
            (*refcount)++;
        }
    }

    template<typename T>
    void *BufferNew<T>::getObj() const {
        return (void *) space_start;
    }


    template<typename T>
    NumN *BufferNew<T>::getRefCount() const {
        return refcount;
    }

    template<typename T>
    int BufferNew<T>::setDataShallow(const BufferNew<T> &autoBuffer) {
        setSpace(autoBuffer.getSpace());
        nn = autoBuffer.size();
        return 1;
    }

    template<typename T>
    int BufferNew<T>::setDataDeep(const BufferNew<T> &autoBuffer) {
        if (size() != autoBuffer.size()) {
            setSize(autoBuffer.size());
        }
        dataCopy(autoBuffer.getObj());
        return 1;
    }

    template<typename T>
    void BufferNew<T>::ensureIndependence() {
        if (!isEmpty() && !isIndependent()) {
            const void *data = space_start;
            NumN nn = size();
            clear();
            setSize(nn);
            dataCopy(data);
        }
    }

    template<typename T>
    bool BufferNew<T>::isEmpty() const {
        if (this->space_address == 0) {
            MATH21_ASSERT(refcount == 0 && space_start == 0,
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

    template<typename T>
    std::string BufferNew<T>::getClassName() const {
        return "BufferNew";
    }

    template<typename T>
    void BufferNew<T>::log(const char *s) const {
        log(std::cout, s);
    }

    template<typename T>
    void BufferNew<T>::log(std::ostream &io, const char *s) const {
        if (s == 0) {
            s = name.c_str();
        }
        io << "BufferNew " << s << ":\n";
        getSpace().log();
    }

    template<typename T>
    void BufferNew<T>::swap(BufferNew<T> &B) {
        m21_swap(space_address, B.space_address);
        m21_swap(space_start, B.space_start);
        m21_swap(space_size, B.space_size);
        m21_swap(nn, B.nn);
        m21_swap(refcount, B.refcount);
    }

}