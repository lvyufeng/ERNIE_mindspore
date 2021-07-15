# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import collections
import gzip
import six
import re
import argparse
import numpy as np
import random
import jieba
from opencc import OpenCC
import os
from mindspore.mindrecord import FileWriter
from src.tokenizer import convert_to_unicode, FullTokenizer

class ErnieDataReader(object):
    def __init__(self,
                 file_list,
                 vocab_path,
                 short_seq_prob,
                 masked_lm_prob,
                 max_predictions_per_seq,
                 dupe_factor,
                 max_seq_len=512,
                 random_seed=1,
                 do_lower_case=True,
                 generate_neg_sample=False):
        # short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words

        self.vocab = self.load_vocab(vocab_path)
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)

        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor

        self.file_list = file_list

        self.max_seq_len = max_seq_len
        self.generate_neg_sample = generate_neg_sample
                
        self.global_rng = random.Random(random_seed)

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().split(";")
        assert len(line) == 5, \
                "One sample must have %d fields!" % 5

        (token_ids, sent_ids, pos_ids, seg_labels, label) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]
    
        assert len(token_ids) == len(sent_ids) == len(pos_ids) == len(
            seg_labels
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label, seg_labels]

    def read_file(self, file):
        assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        with gzip.open(file, "rb") as f:
            for line in f:
                line = line.decode('utf8')
                parsed_line = self.parse_line(
                    line, max_seq_len=self.max_seq_len)
                if parsed_line is None:
                    continue
                yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples
            Args:
                pos_samples: list of positive samples
            
            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            token_seq, type_seq, pos_seq, label, seg_labels = sample
            sep_index = token_seq.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_seq[1:sep_index], seg_labels[1:sep_index])
            else:
                return [
                    token_seq[sep_index + 1:-1], seg_labels[sep_index + 1:-1]
                ]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id)
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens), self.sep_id)

            token_seq = [self.cls_id] + left_tokens + [self.sep_id] + \
                    right_tokens + [self.sep_id]
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) +
                                                             1)
            pos_seq = range(len(token_seq))
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [
                -1
            ]

            assert len(token_seq) == len(type_seq) == len(pos_seq) == len(seg_label_seq), \
                    "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append([token_seq, type_seq, pos_seq, 0, seg_label_seq])

        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples
            
            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]
            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))
    
    def shuffle_samples(self, sample_generator, buffer=1000):
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(samples) == 0:
                yield None
            else:
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    def data_generator(self):
        """
        data_generator
        """
                    
        for index, file_ in enumerate(self.files):
            file_, mask_word_prob = file_.strip().split("\t")
            mask_word = (np.random.random() < float(mask_word_prob))
            if mask_word:
                self.mask_type = "mask_word"
            else:
                self.mask_type = "mask_char"

            sample_generator = self.read_file(file_)
            if self.generate_neg_sample:
                sample_generator = self.mixin_negtive_samples(
                    sample_generator)
            else:
                #shuffle buffered sample
                sample_generator = self.shuffle_samples(
                    sample_generator)

            for sample in sample_generator:
                if sample is None:
                    continue
                sample.append(mask_word)
                yield sample

    def file_based_convert_examples_to_features(self, input_file, output_file, shard_num):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        # "input_ids", "input_mask", "segment_ids", "next_sentence_labels", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"
        examples = self.data_generator()

        writer = FileWriter(file_name=output_file, shard_num=shard_num)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "token_type_id": {"type": "int64", "shape": [-1]},
            "label_ids": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        data = []
        index = 0
        for example in examples:
            if index % 1000 == 0:
                print("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            sample = {
                "input_ids": np.array(record.input_ids, dtype=np.int64),
                "input_mask": np.array(record.input_mask, dtype=np.int64),
                "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                "label_ids": np.array([record.label_id], dtype=np.int64),
            }
            data.append(sample)
            index += 1
        writer.write_raw_data(data)
        writer.commit()

    def create_training_instances(self):
        """Create `TrainingInstance`s from raw text."""
        all_documents = [[]]
        all_documents_segs = [[]]
        p1 = re.compile('<doc (.*)>')
        p2 = re.compile('</doc>')
        cc = OpenCC('t2s')
        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        count = 0
        for input_file in self.file_list:
            count += 1
            with open(input_file, "r") as reader:
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    if p2.match(line):
                        all_documents.append([])
                        all_documents_segs.append([])
                    line = p1.sub('', line)
                    line = p2.sub('', line)
                    line = cc.convert(line)
                    line = convert_to_unicode(line)
                    line = line.strip()

                    segs = self.get_word_segs(line)
                    all_tokens = []
                    all_seg_labels = []
                    for seg in segs:                    
                        tokens = self.tokenizer.tokenize(seg)
                        if len(tokens) > 1:
                            seg_labels = [0] + [1] * (len(tokens) - 1)
                        elif len(tokens) == 1:
                            seg_labels = [0]
                        else:
                            seg_labels= []
                        assert len(tokens) == len(seg_labels)
                        all_tokens.extend(tokens)
                        all_seg_labels.extend(seg_labels)
                    if all_tokens:
                        all_documents[-1].append(all_tokens)
                        all_documents_segs[-1].append(all_seg_labels)
            print(count)
        # Remove empty documents
        all_documents = [x for x in all_documents if x]
        all_documents_segs = [x for x in all_documents_segs if x]

        vocab_words = list(self.tokenizer.vocab.keys())
        instances = []
        for _ in range(self.dupe_factor):
            for document_index in range(len(all_documents)):
                print(document_index)
                # instances.extend(
                #     self.create_instances_from_document(
                #         all_documents, document_index, short_seq_prob,
                #         masked_lm_prob, max_predictions_per_seq, vocab_words))

        self.global_rng.shuffle(instances)
        return instances

    def create_instances_from_document(self,):
        pass

    def get_word_segs(self, sentence):
        segs = jieba.lcut(sentence)
        return segs

def get_file_list(input_file):
    if os.path.isdir(input_file):
        file_list = []
        for root, dirs, files in os.walk(input_file):
            for f in files:
                file_list.append(os.path.join(root, f))
        return file_list
    elif os.path.isfile(input_file):
        return [input_file]
    else:
        raise ValueError('The input path is not a folder or file.')

def main():
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--vocab_path", type=str, default="pretrain_models/converted/vocab.txt", help="vocab file")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--do_lower_case", type=str, default="true",
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed number")
    parser.add_argument("--short_seq_prob", type=float, default=0.1, help="random seed number")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="random seed number")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20, help="random seed number")
    parser.add_argument("--dupe_factor", type=int, default=10, help="random seed number")

    parser.add_argument("--generate_neg_sample", type=str, default="true", help="random seed number")

    parser.add_argument("--input_file", type=str, default="data/text/AA/wiki_00", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    parser.add_argument("--shard_num", type=int, default=0, help="output file shard number")
    args_opt = parser.parse_args()

    file_list = get_file_list(args_opt.input_file)
    reader = ErnieDataReader(
                            file_list=file_list,
                            vocab_path=args_opt.vocab_path,
                            short_seq_prob=args_opt.short_seq_prob,
                            masked_lm_prob=args_opt.masked_lm_prob,
                            max_predictions_per_seq=args_opt.max_predictions_per_seq,
                            dupe_factor=args_opt.dupe_factor,
                            max_seq_len=args_opt.max_seq_len,
                            random_seed=args_opt.random_seed,
                            do_lower_case=True if args_opt.do_lower_case == 'true' else False,
                            generate_neg_sample=True if args_opt.generate_neg_sample == 'true' else False
    )

    reader.create_training_instances()
    # reader.file_based_convert_examples_to_features(input_file=args_opt.input_file,
    #                                                output_file=args_opt.output_file,
    #                                                shard_num=args_opt.shard_num,
    #                                                is_training=True if args_opt.is_training == 'true' else False)
       

if __name__ == "__main__":
    main()