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

#include "inner_header.h"

namespace math21 {
    namespace rl {
        namespace tic_tac_toe {

            struct State {
            private:
                MatZ M;
                NumZ h; // hash value
                // winner
                NumZ w;

                NumZ end;

                void init();

            public:

                void copyTo(State &B) const;

                State();

                // Copy constructor
                State(const State &B);

                virtual ~State();

                const MatZ &getM() const;

                // clear is used in program level.
                void clear();

                // reset is used in algorithm level.
                void reset();

                NumZ hashValue();

                NumB isEnd();

                // symbol sb.
                void nextState(NumN i, NumN j, NumZ sb, State &s_prime) const;

                void log(const char *name = 0) const;

                void log(std::ostream &io, const char *name = 0) const;

                NumZ getWinner() const;

                void serialize(std::ostream &io, SerializeNumInterface &sn) const;

                void deserialize(std::istream &io, DeserializeNumInterface &sn);

            };

            std::ostream &operator<<(std::ostream &io, const State &s);

            struct Player {

            public:
                Player() {}

                virtual ~Player() {}

                virtual void reset() = 0;

                virtual void setState(const State &s) = 0;

                // get V(S) initially.
                virtual void setSymbol(NumZ sb0) = 0;

                virtual NumZ getSymbol() const = 0;

                virtual void backup() = 0;

                // choose a based on s
                virtual void act(VecZ &a, NumB isPrintState=0) = 0;

                virtual void savePolicy() = 0;

                virtual NumB loadPolicy() = 0;

                // proclamation
                virtual void beforePlay(NumB isPrintState=0) = 0;
            };

            struct AIPlayer : public Player {

            private:

                Dict <NumZ, NumR> V;
                NumR alpha;
                NumR epsilon;
                Seqce <State> St;
                Seqce <NumB> g;
                NumZ sb;

                void init();

            public:

                AIPlayer(NumR alpha = 0.1, NumR epsilon = 0.1);

                AIPlayer(const AIPlayer &p) {
                    MATH21_ASSERT(0)
                }

                virtual ~AIPlayer();

                void clear();

                void reset();

                void setState(const State &s);

                // get V(S) initially.
                void setSymbol(NumZ sb0);

                NumZ getSymbol() const;

                void backup();

                // choose a based on s
                void act(VecZ &a, NumB isPrintState=0);

                void savePolicy();

                NumB loadPolicy();

                void beforePlay(NumB isPrintState=0);

            };

            struct HumanPlayer : public Player {

            private:
                NumZ sb;
                Seqce<char> keys;
                State s;

                void init();

                void logKeys() const;

            public:

                HumanPlayer();

                virtual ~HumanPlayer();

                void clear();

                void reset();

                void setState(const State &s0);

                // get V(S) initially.
                void setSymbol(NumZ sb0);

                NumZ getSymbol() const;

                void backup();

                // choose a based on s
                void act(VecZ &a, NumB isPrintState=0);

                void savePolicy();

                NumB loadPolicy();

                void beforePlay(NumB isPrintState=0);
            };

            struct Judger {

            private:
                Player &p1;
                Player &p2;

                void init();

            public:

                Judger(Player &p1, Player &p2);

                virtual ~Judger();

                void clear();

                void reset();

                Player &alternate(NumN &index);

                NumZ play(NumB isPrintState = 0);

                void beforePlay(NumB isPrintState=0);
            };
        }
    }

    void math21_rl_chapter01_tic_tac_toe(NumN nr = 2, NumN nc = 3, NumN epochs = (NumN) 1e2, NumN first_or_second = 2);
}