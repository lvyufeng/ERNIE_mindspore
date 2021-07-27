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

'''
postprocess script.
'''

import os
import json
import argparse
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy, F1, SpanF1
from src.finetune_eval_config import ernie_net_cfg

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--task_type", type=str, default="false",
                    choices=["msra_ner", "chnsenticorp", "xnli", "dbqa", "drcd", "cmrc"],
                    help="Eval task type, default is msra_ner")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
parser.add_argument("--num_class", type=int, default=1, help="number of class")
parser.add_argument("--label_map_config", type=str, default="", help="Label map file path")

args, _ = parser.parse_known_args()

def eval_result_print(assessment_method, callback=None):
    pass

if __name__ == "__main__":
    num_class = args.num_class
    if args.task_type == 'chnsenticorp':
        ernie_net_cfg.seq_length = 256
        assessment_method = 'accuracy'
        callback = Accuracy()
    elif args.task_type == 'xnli':
        ernie_net_cfg.seq_length = 512
        assessment_method = 'accuracy'
        callback = Accuracy()
    elif args.task_type == 'dbqa':
        ernie_net_cfg.seq_length = 512
        assessment_method = 'f1'
        callback = F1(num_class)
    elif args.task_type == 'msra_ner':
        ernie_net_cfg.seq_length = 512
        with open(args.label_map_config) as f:
            tag_to_index = json.load(f)
        assessment_method = 'spanf1'
        callback = SpanF1(num_class)
    elif args.task_type == 'drcd':
        ernie_net_cfg.seq_length = 512
        assessment_method = 'mrc'
    elif args.task_type == 'cmrc':
        ernie_net_cfg.seq_length = 512
        assessment_method = 'mrc'
    else:
        raise ValueError("Unsupported task type.")

    file_name = os.listdir(args.label_dir)
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32).reshape(ernie_net_cfg.seq_length * args.batch_size, num_class)
        logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, ernie_net_cfg.seq_length))
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")
