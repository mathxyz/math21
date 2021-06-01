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

#include "Digraph.h"

namespace math21 {
    namespace data_structure {

            /*
             * *  Determine single-source or multiple-source reachability in a digraph
             *  using depth first search.
             *  Runs in O(E + V) time.
             * */

            class DirectedDFS {
                    private:
                    bool *marked;  // marked[v] = true iff v is reachable from source(s)
                    int _count;         // number of vertices reachable from source(s)
                    int V;

                    void dfs(const Digraph &G, int v) {
                        if (marked[v]) {
                            return;
                        }
                        _count++;
                        marked[v] = true;
                        auto i = G.adjacent(v);
                        while (i.hasNext()) {
                            auto w = i.next();
                            if (!marked[w]) dfs(G, w);
                        }
                    }

                    void validateVertex(int v) const {
                        if (v < 0 || v >= V) {
                            MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V - 1))
                        }
                    }

                    void validateVertices(ListIterator<int> vertices) const {
                        while (vertices.hasNext()) {
                            auto v = vertices.next();
                            if (v < 0 || v >= V) {
                                MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V - 1))
                            }
                        }
                    }

                    public:
                    // Computes the vertices in digraph G that are reachable from the source vertex s.
                    DirectedDFS(const Digraph &G, int s) {
                        V = G.V();
                        marked = new bool[V];
                        for (int v = 0; v < V; ++v) {
                            marked[v] = false;
                        }
                        validateVertex(s);
                        dfs(G, s);
                    }

                    virtual ~DirectedDFS() {
                        delete[]marked;
                        V = 0;
                        _count = 0;
                    }

                    // Computes the vertices in digraph {@code G} that are
                    // connected to any of the source vertices {@code sources}.
                    DirectedDFS(const Digraph &G, ListIterator<int> sources) {
                        V = G.V();
                        marked = new bool[V];
                        for (int v = 0; v < V; ++v) {
                            marked[v] = false;
                        }
                        validateVertices(sources);
//                        int j = 0;
                        while (sources.hasNext()) {
                            auto v = sources.next();
                            if (!marked[v]) dfs(G, v);
                        }
                    }

                    // Is there a directed path from the source vertex (or any
                    // of the source vertices) and vertex {@code v}?
                    bool isMarked(int v) const {
                        validateVertex(v);
                        return marked[v];
                    }

                    // Returns the number of vertices reachable from the source vertex
                    // (or source vertices).
                    int count() const {
                        return _count;
                    }

                    void log(const char *name = 0) const {
                        log(std::cout, name);
                    }

                    void log(std::ostream &io, const char *name = 0) const {
                        if (name) {
                            io << "DirectedDFS: " << name << "\n";
                            io << V << " vertices\n";
                            io << _count << " count\n";
                        }
                        for (int v = 0; v < V; v++) {
                            io << v << ": " << math21_string_to_string(isMarked(v)) << "\n";
                        }
                    }
            };
    }
}