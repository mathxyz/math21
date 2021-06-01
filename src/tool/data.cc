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

#include "../point/files.h"
#include "../op/files.h"
#include "text.h"
#include "data.h"

namespace math21 {

    void math21_data_text_build_dataset(const char *filePath, TenR &seqs,
                                        NumN sequence_length,
                                        NumN max_lines,
                                        NumN alphabet_size, const char *savePath) {
        Seqce<TenN8> lines;
        math21_file_text_read_lines(filePath, lines);
        if (max_lines == 0) {
            max_lines = lines.size();
        }
        NumN n = xjmin(max_lines, lines.size());
        NumN n_lines = 0;
        NumN thr = 1;
//    NumN thr = 0;
        for (NumN i = 1; i <= n; ++i) {
            if (lines(i).size() > thr) {
                ++n_lines;
            }
        }

        seqs.setSize(sequence_length, n_lines, alphabet_size);
        VecR padded_line;
        padded_line.setSize(sequence_length);
        MatR letters(1, alphabet_size);
        letters.letters(0);
        MatR oneHotMatrix;
        for (NumN i = 1, j = 1; i <= n; ++i) {
            const TenN8 &line = lines(i);
            if (line.size() > thr) {
                padded_line = ' ';
                math21_op_vector_set_by_vector(line, padded_line);
                math21_op_ele_is_equal(padded_line, letters, oneHotMatrix); // string to one hot
                math21_op_tensor_sub_axis_i_expand_and_set(oneHotMatrix, seqs, j - 1, 2);
                ++j;
            }
        }
        if (savePath) {
            math21_io_save(savePath, seqs);
        }
    }
}