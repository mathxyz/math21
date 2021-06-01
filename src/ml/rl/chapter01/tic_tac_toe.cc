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

#include "inner.h"
#include "tic_tac_toe.h"

namespace math21 {
    namespace rl {
        namespace tic_tac_toe {
            // nr = nc
            // BOARD_ROWS
            NumN paras_nr = 3;
            // BOARD_COLS
            NumN paras_nc = 2;
            // BOARD_SIZE
            NumN paras_size = paras_nr * paras_nc;

            NumN paras_epochs = 100;

            std::string paras_name;

            std::string paras_border;

            DefaultRandomEngine engine(21);

            Dict<NumZ, State> S;

            //////// State
            const MatZ &State::getM() const {
                return M;
            }

            void State::init() {
                M.setSize(paras_nr, paras_nc);
                M = 0;
                h = -1;
                w = 0;
                end = -1;
            }


            void State::copyTo(State &B) const {
                B.reset();
                B.M.assign(M);
                B.h = h;
                B.w = w;
                B.end = end;
            }

            State::State() {
                init();
            }

            // Copy constructor
            State::State(const State &B) {
//                MATH21_ASSERT(0, "State Copy constructor.");
                B.copyTo(*this);
            }

            State::~State() {
                clear();
            }

            void State::clear() {
            }

            // reset
            void State::reset() {
                init();
            }

            NumZ State::hashValue() {
                if (h != -1) {
                    return h;
                }
                NumN x = 0;
                for (NumN i = 1; i <= M.nrows(); ++i) {
                    for (NumN j = 1; j <= M.ncols(); ++j) {
                        x = x * 3 + (M(i, j) + 1);
                    }
                }
                h = x;
                return h;
            }

            NumB State::isEnd() {
                if (end != -1) {
                    return (NumB) end;
                }
                SequenceZ re;
                for (NumN i = 1; i <= M.nrows(); ++i) {
                    re.add(math21_operator_matrix_sum_row_i(M, i));
                }
                for (NumN i = 1; i <= M.ncols(); ++i) {
                    re.add(math21_operator_matrix_sum_col_i(M, i));
                }

                // check diagonal if M is square
                if (M.nrows() == M.ncols()) {
                    re.add(math21_operator_matrix_trace(M));
                    re.add(math21_operator_matrix_reverse_trace(M));
                }

                NumZ x0;
                for (NumN i = 1; i <= re.size(); ++i) {
                    NumZ x = re(i);
                    if (i <= M.nrows()) {
                        x0 = paras_nc;
                    } else {
                        x0 = paras_nr;
                    }
                    if (x == x0) {
                        w = 1;
                        end = 1;
                        return (NumB) end;
                    }
                    if (x == -x0) {
                        w = -1;
                        end = 1;
                        return (NumB) end;
                    }
                }

                //# whether it's a tie
                NumZ sv = (NumZ) math21_operator_norm(M, 1);
                if (sv == paras_size) {
                    w = 0;
                    end = 1;
                    return (NumB) end;
                }

                //# game is still going on
                end = 0;
                return (NumB) end;
            }

            // symbol sb.
            void State::nextState(NumN i, NumN j, NumZ sb, State &s_prime) const {
                s_prime.reset();
                s_prime.M.assign(M);
                MATH21_ASSERT(s_prime.M(i, j) == 0, "input error!")
                s_prime.M(i, j) = sb;
            }

            void State::log(const char *name) const {
                log(std::cout, name);
            }

            void State::log(std::ostream &io, const char *name) const {
                if (name) {
                    io << "State " << name << "\n";
                    io << "isEnd: " << end << "\n";
                    io << "h: " << h << "\n";
                }
                for (NumN i = 1; i <= M.nrows(); ++i) {
                    io << paras_border << "\n";
                    std::string out = "| ";
                    for (NumN j = 1; j <= M.ncols(); ++j) {
                        std::string token;
                        if (getM()(i, j) == 1) {
                            token = "O";
                        } else if (getM()(i, j) == -1) {
                            token = "X";
                        } else {
                            token = " ";
                        }
                        out += token + " | ";
                    }
                    io << out << "\n";
                }
                io << paras_border << "\n";
            }

            NumZ State::getWinner() const {
                return w;
            }

            void State::serialize(std::ostream &io, SerializeNumInterface &sn) const {
                math21_io_serialize(io, M, sn);
                math21_io_serialize(io, h, sn);
                math21_io_serialize(io, w, sn);
                math21_io_serialize(io, end, sn);
            }

            void State::deserialize(std::istream &io, DeserializeNumInterface &sn) {
                clear();
                math21_io_deserialize(io, M, sn);
                math21_io_deserialize(io, h, sn);
                math21_io_deserialize(io, w, sn);
                math21_io_deserialize(io, end, sn);
            }

            std::ostream &operator<<(std::ostream &io, const State &s) {
                s.log(io);
                return io;
            }

            ////////

            void math21_io_serialize(std::ostream &io, const State &m, SerializeNumInterface &sn) {
                m.serialize(io, sn);
            }

            void math21_io_deserialize(std::istream &io, State &m, DeserializeNumInterface &sn) {
                m.deserialize(io, sn);
            }

            ////////

            void saveStates(const Dict<NumZ, State> &S) {
                std::string s = "states.bin";

                std::ofstream out;
                out.open(s, std::ofstream::binary);
                SerializeNumInterface_simple sn;

                math21_io_serialize(out, paras_name, sn);
                math21_io_serialize(out, paras_nr, sn);
                math21_io_serialize(out, paras_nc, sn);
                math21_io_serialize(out, S, sn);
                out.close();
                m21vlog("There are %d states.\n", S.size());
                m21log("All states saved.");
            }

            NumB loadStates(Dict<NumZ, State> &S) {
                std::string s = "states.bin";

                std::ifstream in;
                in.open(s, std::ifstream::binary);
                if (!in.is_open()) {
                    std::cerr << "open " << s << " fail!\n";
                    return 0;
                }
                DeserializeNumInterface_simple sn;
                NumN r, c;
                std::string name0;
                math21_io_deserialize(in, name0, sn);
                math21_io_deserialize(in, r, sn);
                math21_io_deserialize(in, c, sn);
                if (!(name0 == paras_name && r == paras_nr && c == paras_nc)) {
                    return 0;
                }
                m21log("load all states.");
                math21_io_deserialize(in, S, sn);
                in.close();
                return 1;
            }

            void get_all_states_impl(const State &s, NumZ cs, Dict<NumZ, State> &S) {
                for (NumN i = 1; i <= paras_nr; ++i) {
                    for (NumN j = 1; j <= paras_nc; ++j) {
                        if (s.getM()(i, j) == 0) {
                            State s_prime;
                            s.nextState(i, j, cs, s_prime);
                            NumZ h_prime = s_prime.hashValue();
                            if (!S.has(h_prime)) {
                                S.add(h_prime, s_prime);
                                if (S.size() % 10000 == 0) {
                                    m21vlog("Get %d states.\n", S.size());
                                }
                                if (!s_prime.isEnd()) {
                                    get_all_states_impl(s_prime, -cs, S);
                                }
                            }
                        }
                    }
                }
            }

            void get_all_states(Dict<NumZ, State> &S) {
                if (loadStates(S)) {
                    return;
                }
                m21log("Calculating all states.");
                m21vlog("Max search %d times\n", (NumN) m21_pow(3, paras_size));
                NumZ cs = 1;
                State s;
                S.clear();
                S.add(s.hashValue(), s);
                get_all_states_impl(s, cs, S);
                saveStates(S);
            }

            //////// AIPlayer
            void AIPlayer::init() {
            }


            AIPlayer::AIPlayer(NumR alpha, NumR epsilon) : alpha(alpha), epsilon(epsilon) {
                V.clear();
                St.clear();
                g.clear();
                sb = 0;
            }

            AIPlayer::~AIPlayer() {
                clear();
            }

            void AIPlayer::clear() {
            }

            void AIPlayer::reset() {
                St.clear();
                g.clear();
            }

            void AIPlayer::setState(const State &s) {
                St.push(s);
                g.push(1);
            }

            NumZ AIPlayer::getSymbol() const {
                return sb;
            }

            // get V(S) initially.
            void AIPlayer::setSymbol(NumZ sb0) {
                sb = sb0;
                NumN i;
                NumN n = S.size();
                NumZ h;
                for (i = 1; i <= n; ++i) {
                    h = S.keyAtIndex(i);
                    State &s = S.valueAtIndex(i);
                    if (s.isEnd()) {
                        if (s.getWinner() == sb) {
                            V.add(h, 1);
                        } else if (s.getWinner() == 0) {
                            V.add(h, 0.5);
                        } else {
                            V.add(h, 0);
                        }
                    } else {
                        V.add(h, 0.5);
                    }
                }
            }

            void AIPlayer::backup() {
                NumN i, n;
                n = St.size();
                NumZ s, s_prime;
                NumR delta;
                for (i = n - 1; i >= 1; --i) {
                    s = St(i).hashValue();
                    s_prime = St(i + 1).hashValue();
                    NumR &V_s_prime = V.valueAt(s_prime);
                    NumR &V_s = V.valueAt(s);
                    delta = g(i) * (V_s_prime - V_s);
                    V_s = V_s + alpha * delta;
                }
            }

            // choose a based on s
            void AIPlayer::act(VecZ &a0, NumB isPrintState) {
                MATH21_ASSERT(a0.isSameSize(3))
                State &s = St.at(St.size());
                Seqce<VecN> as;
                Seqce<NumZ> ss;
                VecN a(2);
                State s_prime;
                for (NumN i = 1; i <= paras_nr; ++i) {
                    for (NumN j = 1; j <= paras_nc; ++j) {
                        if (s.getM()(i, j) == 0) {
                            a = i, j;
                            as.push(a);
                            s.nextState(i, j, sb, s_prime);
                            ss.push(s_prime.hashValue());
                        }
                    }
                }
                NumR r;
                RanUniform ran(engine);
                math21_random_draw(r, ran);
                if (r < epsilon) {
                    NumN i;
                    ran.set(1, as.size());
                    math21_random_draw(i, ran);
                    a0 = as(i)(1), as(i)(2), sb;
                    g.at(g.size()) = 0;
                    return;
                }

                SeqceR Vt(ss.size());
                for (NumN i = 1; i <= ss.size(); ++i) {
                    Vt.at(i) = V.valueAt(ss(i));
                }
                NumN i = math21_operator_container_argmax_random(Vt, engine);
                a0 = as(i)(1), as(i)(2), sb;
                if (isPrintState) {
                    m21log("AI player:");
                }
            }

            void AIPlayer::savePolicy() {
                std::ostringstream oss;
                oss << "policy_" << (sb == 1 ? "first" : "second") << ".bin";
                std::string s = oss.str();

                std::ofstream out;
                out.open(s, std::ofstream::binary);
                SerializeNumInterface_simple sn;
                math21_io_serialize(out, paras_name, sn);
                math21_io_serialize(out, paras_nr, sn);
                math21_io_serialize(out, paras_nc, sn);
                math21_io_serialize(out, paras_epochs, sn);
                math21_io_serialize(out, V, sn);
                out.close();

            }

            NumB AIPlayer::loadPolicy() {
                std::ostringstream oss;
                oss << "policy_" << (sb == 1 ? "first" : "second") << ".bin";
                std::string s = oss.str();

                std::ifstream in;
                in.open(s, std::ifstream::binary);
                if (!in.is_open()) {
                    std::cerr << "open " << s << " fail!\n";
                    return 0;
                }
                DeserializeNumInterface_simple sn;
                std::string name0;
                NumN r, c, ep;
                math21_io_deserialize(in, name0, sn);
                math21_io_deserialize(in, r, sn);
                math21_io_deserialize(in, c, sn);
                math21_io_deserialize(in, ep, sn);
                if (!(name0 == paras_name && r == paras_nr && c == paras_nc && ep == paras_epochs)) {
                    return 0;
                }
                math21_io_deserialize(in, V, sn);
                in.close();
                return 1;
            }

            void AIPlayer::beforePlay(NumB isPrintState) {
                NumN first_or_second = sb == 1 ? 1 : 2;
                if (isPrintState) {
                    m21vlog("AI player %d is ready!\n", first_or_second);
                }
            }

            //////// Human player


            void HumanPlayer::init() {
                sb = 0;
                keys.setSize(paras_size);
                Tensor<char> M(5, 5);
                M =
                        '6', '7', '8', '9', '0',
                        '1', '2', '3', '4', '5',
                        'q', 'w', 'e', 'r', 't',
                        'a', 's', 'd', 'f', 'g',
                        'z', 'x', 'c', 'v', 'b';
                MATH21_ASSERT((xjIsIn(paras_nr, 1, 5)), "not support in current version!")
                MATH21_ASSERT((xjIsIn(paras_nc, 1, 5)), "not support in current version!")

                math21_operator_matrix_reverse_x_axis(M);

                VecN row_indexes(paras_nr);
                row_indexes.letters();
                VecN col_indexes(paras_nc);
                col_indexes.letters();

                Tensor<char> K;
                math21_operator_matrix_submatrix(M, K, row_indexes, col_indexes);
                math21_operator_matrix_reverse_x_axis(K);

                math21_operator_setData_to_container(K, keys);
            }


            void HumanPlayer::logKeys() const {
                Tensor<char> M(paras_nr, paras_nc);
                math21_operator_setData_by_container(M, keys);
                m21log("all keys:");
                M.log(0, 0, 0);
            }

            HumanPlayer::HumanPlayer() {
                init();
            }

            HumanPlayer::~HumanPlayer() {

            }

            void HumanPlayer::clear() {

            }

            void HumanPlayer::reset() {

            }

            void HumanPlayer::setState(const State &s0) {
                s0.copyTo(s);
            }

            // get V(S) initially.
            void HumanPlayer::setSymbol(NumZ sb0) {
                sb = sb0;
            }

            NumZ HumanPlayer::getSymbol() const {
                return sb;
            }

            void HumanPlayer::backup() {

            }

            // choose a based on s
            void HumanPlayer::act(VecZ &a, NumB isPrintState) {
                MATH21_ASSERT(a.isSameSize(3))
                char key;
                while (1) {
                    m21log("Your turn, input:");
                    m21input(key);
                    NumN n = math21_operator_container_index(keys, key);
                    if (n == 0) {
                        m21log("Unknown key!");
                        logKeys();
                        continue;
                    }
                    NumN i = (n - 1) / paras_nc + 1;
                    NumN j = (n - 1) % paras_nc + 1;
                    a = i, j, sb;
                    if (s.getM()(i, j) == 0) {
                        break;
                    } else {
                        m21log("This is taken.");
                    }
                }
                if (isPrintState) {
                    m21log("You:");
                }
            }

            void HumanPlayer::savePolicy() {

            }

            NumB HumanPlayer::loadPolicy() {
                return 1;
            }

            void HumanPlayer::beforePlay(NumB isPrintState) {
                if (isPrintState) {
                    m21log("Are you ready?");
                    logKeys();
                }
            }

            //////// Judger
            void Judger::init() {
                NumZ sb1;
                NumZ sb2;
                sb1 = 1;
                sb2 = -1;
                p1.setSymbol(sb1);
                p2.setSymbol(sb2);
            }


            Judger::Judger(Player &p1, Player &p2) : p1(p1), p2(p2) {
                init();
            }

            Judger::~Judger() {
                clear();
            }

            void Judger::clear() {

            }

            void Judger::reset() {
                p1.reset();
                p2.reset();
            }

            Player &Judger::alternate(NumN &index) {
                NumN n = 2;
                MATH21_ASSERT(index < n)
                ++index;
                NumN i = index;
                index = index % n;
                switch (i) {
                    case 1:
                        return p1;
                    case 2:
                        return p2;
                    default: MATH21_ASSERT(0)
                }
                return p1;
            }

            void Judger::beforePlay(NumB isPrintState) {
                p1.beforePlay(isPrintState);
                p2.beforePlay(isPrintState);
            }

            NumZ Judger::play(NumB isPrintState) {
                reset();
                State s0, s_prime;
                p1.setState(s0);
                p2.setState(s0);
                if (isPrintState) {
                    m21log("Initial state.");
                    s0.log();
                }
                NumN index = 0;
                VecZ a(3);
                NumZ h_prime;
                State *ps = &s0;
                while (1) {
                    State &s = *ps;
                    Player &p = alternate(index);
                    p.act(a, isPrintState);
                    s.nextState((NumN) a(1), (NumN) a(2), a(3), s_prime);
                    h_prime = s_prime.hashValue();
                    State &s_new = S.valueAt(h_prime);
                    p1.setState(s_new);
                    p2.setState(s_new);
                    if (isPrintState) {
                        s_new.log();
                    }
                    if (s_new.isEnd()) {
                        return s_new.getWinner();
                    }
                    ps = &s_new;
                }
            }

            void train(NumN epochs, NumN print_every_n = 500) {
                m21log("Training two AI players.");
                AIPlayer _p1 = AIPlayer(0.1, 0.01);
                AIPlayer _p2 = AIPlayer(0.1, 0.01);
                Player &p1 = _p1;
                Player &p2 = _p2;
                Judger j = Judger(p1, p2);
                j.beforePlay(0);
                NumR p1_win = 0;
                NumR p2_win = 0;
                NumN i;
                NumZ w;
                for (i = 1; i <= epochs; ++i) {
                    w = j.play(0);
                    if (w == 1) {
                        p1_win = p1_win + 1;
                    }
                    if (w == -1) {
                        p2_win = p2_win + 1;
                    }
                    if (i % print_every_n == 0) {
                        m21vlog("Epoch %d, player 1 win %.02f, win rate: %.02f, player 2 win %.02f, win rate: %.02f.\n",
                               i, p1_win, p1_win / i, p2_win, p2_win / i);
                    }
                    p1.backup();
                    p2.backup();
                    j.reset();
                }
                p1.savePolicy();
                p2.savePolicy();
                m21log("Training finish.");
            }

            void compete(NumN turns) {
                m21log("Now Two AI players compete.");
                AIPlayer _p1 = AIPlayer(0.1, 0);
                AIPlayer _p2 = AIPlayer(0.1, 0);
                Player &p1 = _p1;
                Player &p2 = _p2;
                Judger j = Judger(p1, p2);
                p1.loadPolicy();
                p2.loadPolicy();

                j.beforePlay(1);

                NumR p1_win = 0;
                NumR p2_win = 0;
                NumN i;
                NumZ w;
                for (i = 1; i <= turns; ++i) {
                    w = j.play(0);
                    if (w == 1) {
                        p1_win = p1_win + 1;
                    }
                    if (w == -1) {
                        p2_win = p2_win + 1;
                    }
                    j.reset();
                }
                m21vlog("Compete results: %d turns\n"
                       "player 1 win %.02f, win rate %.02f, player 2 win %.02f, win rate %.02f.\n",
                       turns, p1_win, p1_win / turns, p2_win, p2_win / turns);
            }

            Player &get_player_by_symbol(Player &humanPlayer, Player &aiPlayer, NumZ human_symbol, NumZ symbol) {
                if (symbol == human_symbol) {
                    return humanPlayer;
                } else {
                    return aiPlayer;
                }
            }

            // first_or_second: {1, 2}, human symbol: {-1, 1}
            NumB play(NumN first_or_second = 1) {
                m21log("Start play");
                NumZ human_symbol;
                if (first_or_second == 1) {
                    human_symbol = 1;
                } else {
                    human_symbol = -1;
                }
                HumanPlayer humanPlayer = HumanPlayer();
                AIPlayer aiPlayer = AIPlayer(0.1, 0);

                Player &p1 = get_player_by_symbol(humanPlayer, aiPlayer, human_symbol, 1);
                Player &p2 = get_player_by_symbol(humanPlayer, aiPlayer, human_symbol, -1);

                Judger j = Judger(p1, p2);
                NumB flag;
                flag = p1.loadPolicy();
                if (flag == 0) {
                    return 0;
                }
                flag = p2.loadPolicy();
                if (flag == 0) {
                    return 0;
                }

                j.beforePlay(1);

                NumN i;
                NumZ w;
                for (i = 1; i <= MATH21_MAX; ++i) {
                    w = j.play(1);
                    if (w == human_symbol) {
                        m21log("You win!");
                    } else if (w == -human_symbol) {
                        m21log("You lose!");
                    } else {
                        m21log("It is a tie!");
                    }
                    j.reset();
                    m21pause_pressEnter("Again? Press Enter to continue.");
                }
                return 1;
            }

            void set_paras(NumN nr0, NumN nc0, NumN epochs0) {
                paras_name = "tic_tac_toe";

                paras_nr = nr0;
                paras_nc = nc0;
                paras_size = paras_nr * paras_nc;
                paras_epochs = epochs0;
                if (paras_nc == 2) {
                    paras_border = "---------";
                } else if (paras_nc == 3) {
                    paras_border = "-------------";
                } else if (paras_nc == 4) {
                    paras_border = "-----------------";
                } else {
                    paras_border = "---------------------";
                }
            }

            void rl_tic_tac_toe(NumN nr, NumN nc, NumN epochs, NumN first_or_second) {
                set_paras(nr, nc, epochs);

                get_all_states(S);

                NumB flag = play(first_or_second);
                if (flag == 1) {
                    return;
                }

                train(epochs);
                compete((NumN) 1e3);
                play(first_or_second);
            }

        }
    }

    void math21_rl_chapter01_tic_tac_toe(NumN nr, NumN nc, NumN epochs, NumN first_or_second) {
        rl::tic_tac_toe::rl_tic_tac_toe(nr, nc, epochs, first_or_second);
    }

}