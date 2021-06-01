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

#include "../../tool/files_c.h"
#include "option.h"

void math21_function_option_insert(m21list *l, char *key, char *val) {
    m21kvp *p = math21_vector_malloc_cpu(sizeof(m21kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    math21_data_structure_list_insert(l, p);
}

int math21_function_option_read(char *s, m21list *options) {
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for (i = 0; i < len; ++i) {
        if (s[i] == '=') {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }
    if (i == len - 1) return 0;
    char *key = s;
    math21_function_option_insert(options, key, val);
    return 1;
}

m21list *math21_function_option_cfg_read(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == 0) math21_file_error(filename);
    char *line;
    int nu = 0;
//    list *options = make_list();
    m21list *options = math21_data_structure_list_create();
    m21section *current = 0;
    while ((line = math21_file_get_line_c(file)) != 0) {
        ++nu;
        math21_string_strip(line);
        switch (line[0]) {
            case '[':
                current = math21_vector_malloc_cpu(sizeof(m21section));
                math21_data_structure_list_insert(options, current);
                current->options = math21_data_structure_list_create();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                math21_vector_free_cpu(line);
                break;
            default:
                if (!math21_function_option_read(line, current->options)) {
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    math21_vector_free_cpu(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

void math21_function_option_free_section(m21section *s) {
    math21_vector_free_cpu(s->type);
    m21node *n = s->options->first;
    while (n) {
        m21kvp *pair = (m21kvp *) n->val;
        math21_vector_free_cpu(pair->key);
        math21_vector_free_cpu(pair);
        m21node *next = n->next;
        math21_vector_free_cpu(n);
        n = next;
    }
    math21_vector_free_cpu(s->options);
    math21_vector_free_cpu(s);
}

char *math21_function_option_find(m21list *l, const char *key) {
    m21node *n = l->first;
    while (n) {
        m21kvp *p = (m21kvp *) n->val;
        if (strcmp(p->key, key) == 0) {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

const char *math21_function_option_find_str(m21list *l, const char *key, const char *def) {
    char *v = math21_function_option_find(l, key);
    if (v) return v;
    if (def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

const char *math21_function_option_find_str_quiet(m21list *l, const char *key, const char *def) {
    char *v = math21_function_option_find(l, key);
    if (v) return v;
    return def;
}

int math21_function_option_find_int(m21list *l, const char *key, int def) {
    char *v = math21_function_option_find(l, key);
    if (v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int math21_function_option_find_int_quiet(m21list *l, const char *key, int def) {
    char *v = math21_function_option_find(l, key);
    if (v) return atoi(v);
    return def;
}

NumN math21_function_option_find_NumN_quiet(m21list *l, const char *key, NumN def) {
    char *v = math21_function_option_find(l, key);
    if (v) {
        int value = atoi(v);
        math21_tool_assert(value >= 0);
        return (NumN) value;
    }
    return def;
}

float math21_function_option_find_float_quiet(m21list *l, const char *key, float def) {
    char *v = math21_function_option_find(l, key);
    if (v) return atof(v);
//    if (v) return strtod(v);
    return def;
}

float math21_function_option_find_float(m21list *l, const char *key, float def) {
    char *v = math21_function_option_find(l, key);
    if (v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}

void math21_function_option_log_unused(m21list *l) {
    m21node *n = l->first;
    while (n) {
        m21kvp *p = (m21kvp *) n->val;
        if (!p->used) {
            fprintf(stdout, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}
