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
"""
Data operations, will be used in run_pretrain.py
"""
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C

def create_classification_dataset(batch_size=1,
                                  repeat_count=1,
                                  data_file_path=None,
                                  schema_file_path=None,
                                  do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset([data_file_path],
                              columns_list=["input_ids", "input_mask", "token_type_id", "label_ids"],
                              shuffle=do_shuffle)
    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="token_type_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=False)
    return data_set

def create_ner_dataset(batch_size=1, repeat_count=1, assessment_method="accuracy", data_file_path=None,
                       dataset_format="mindrecord", schema_file_path=None, do_shuffle=True, drop_remainder=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if dataset_format == "mindrecord":
        dataset = ds.MindDataset([data_file_path],
                                 columns_list=["input_ids", "input_mask", "token_type_id", "label_ids"],
                                 shuffle=do_shuffle)
    else:
        dataset = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                     columns_list=["input_ids", "input_mask", "token_type_id", "label_ids"],
                                     shuffle=do_shuffle)
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        dataset = dataset.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        dataset = dataset.map(operations=type_cast_op, input_columns="label_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="token_type_id")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_mask")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_ids")
    dataset = dataset.repeat(repeat_count)
    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset