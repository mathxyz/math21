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
     * Determine reachability in a digraph from a given vertex using
     *  depth-first search.
     *  Runs in O(E + V) time.
     * */
        class DepthFirstDirectedPaths {
        private:
            bool *marked;  // marked[v] = true iff v is reachable from s
            int *edgeTo;      // edgeTo[v] = last edge on path from s to v
            const int s;       // source vertex
            int V;

            void dfs(const Digraph &G, int v) {
                marked[v] = true;
                auto i = G.adjacent(v);
                while (i.hasNext()) {
                    int w = i.next();
                    if (!marked[w]) {
                        edgeTo[w] = v;
                        dfs(G, w);
                    }
                }
            }

            void validateVertex(int v) const {
                if (v < 0 || v >= V) {
                    MATH21_ASSERT(0, "vertex " << v << " is not between 0 and " << (V - 1))
                }
            }

        public:
            DepthFirstDirectedPaths(const Digraph &G, int s) : s(s) {
                V = G.V();
                marked = new bool[V];
                for (int v = 0; v < V; ++v) {
                    marked[v] = false;
                }
                edgeTo = new int[V];
                validateVertex(s);
                dfs(G, s);
            }

            virtual ~DepthFirstDirectedPaths() {
                delete[]marked;
                delete[]edgeTo;
            }

            // Is there a directed path from the source vertex {@code s} to vertex {@code v}?
            bool hasPathTo(int v) const {
                validateVertex(v);
                return marked[v];
            }

            // Returns a directed path from the source vertex {@code s} to vertex {@code v}
            void pathTo(int v, VecZ &p) const;

            void log(const char *name = 0) const {
                log(std::cout, name);
            }

            void logDetail(std::ostream &io, const char *name = 0) const {
                if (name) {
                    io << "DepthFirstDirectedPaths: " << name << "\n";
                    io << "source " << s << "\n";
                    io << V << " vertices\n";
                }
                for (int v = 0; v < V; v++) {
                    io << v << ": " << math21_string_to_string(marked[v]) << "\n";
                }
                for (int v = 0; v < V; v++) {
                    io << v << ": " << math21_string_to_string(edgeTo[v]) << "\n";
                }
            }

            void log(std::ostream &io, const char *name = 0) const;
        };
    }
}