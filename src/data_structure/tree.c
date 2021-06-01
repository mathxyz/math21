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

#include "../tool/files_c.h"
#include "tree_c.h"

m21tree *math21_data_struncture_tree_read(const char *filename) {
    m21tree t = {0};
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = math21_file_get_line_c(fp)) != 0) {
        char *id = math21_vector_calloc_cpu(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = math21_vector_realloc_cpu(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;

        t.child = math21_vector_realloc_cpu(t.child, (n + 1) * sizeof(int));
        t.child[n] = -1;

        t.name = math21_vector_realloc_cpu(t.name, (n + 1) * sizeof(char *));
        t.name[n] = id;
        if (parent != last_parent) {
            ++groups;
            t.group_offset = math21_vector_realloc_cpu(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = math21_vector_realloc_cpu(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = math21_vector_realloc_cpu(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = math21_vector_realloc_cpu(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = math21_vector_realloc_cpu(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = math21_vector_calloc_cpu(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i) t.leaf[i] = 1;
    for (i = 0; i < n; ++i) if (t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    m21tree *tree_ptr = math21_vector_calloc_cpu(1, sizeof(m21tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}
