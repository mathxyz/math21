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
    namespace ad {

        // VariableMap
        struct VariableMap {
        private:
            NumN constant_0;
            NumN constant_1;
            NumN constant_m1;

            Seqce<Variable *> v; // local and global together
            Seqce<Variable *> local; // non-shared
            Seqce<Variable *> global; // shared
            Set &V; // variable ids

            // backup
            Set V_backup;
            NumN local_size_backup;
            NumN global_size_backup;
            NumN v_size_backup;

            void init();

            NumN _createSharedC(NumR x, const char *name);

            void _createSomeSharedC();

            NumN _createV(const char *name, NumB isShared, NumN sharedId);

        public:
            VariableMap(Set &V);

            virtual ~VariableMap();

            void clear();

            // todo: error, should make it different.
            NumN get_constant_0();

            NumN get_constant_1();

            // -1
            NumN get_constant_m1();

            NumN size() const;

            NumB isEmpty() const;

            NumB log(const char *s = 0) const;

            NumB log(std::ostream &io, const char *s = 0) const;

            // this will make previous references to pointer content invalid.
            // So use references to data.
            NumN createV(const char *name = 0);

            // create constant.
            NumN createC(const char *name);

            void setDeviceType(NumN id, NumN deviceType);

            void setValue(NumN id, NumR x);

            Variable &at(NumN i);

            const Variable &operator()(NumN i) const;

            Set &getV();

            void backup();

            void restore();

            void reset();
        };

        // constant X by variable X
        void setSizeCXUXByX(const Set &X, VariableMap &data);

        // constant Y by X
        void setSizeYByX(const Set &X, const Set &Y, VariableMap &data);

        void setSizeyByx(NumN x, NumN y, VariableMap &data);
    }
}