#!/bin/bash
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
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh run_standalone_finetune.sh TASK_TYPE"
    echo "for example: sh convert_dataset.sh msra_ner"
    echo "TASK_TYPE including [msra_ner, chnsenticorp]"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
exit 1
fi

mkdir -p ms_log
mkdir -p save_models
CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models
DATA_PATH=${CUR_DIR}/data
SAVE_PATH=${CUR_DIR}/save_models
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

TASK_TYPE=$1
case $TASK_TYPE in
  "msra_ner")
    TRAIN_BATCH_SIZE=24
    EVAL_BATCH_SIZE=32
    EPOCH_NUM=6
    NUM_CLASS=7
    ;;
  "chnsenticorp")
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=16
    EPOCH_NUM=10
    NUM_CLASS=2
    ;;
  esac
python ${CUR_DIR}/run_ernie_classifier.py  \
    --device_target="Ascend" \
    --do_train="true" \
    --do_eval="true" \
    --device_id=7 \
    --epoch_num=$EPOCH_NUM \
    --num_class=$NUM_CLASS \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --save_finetune_checkpoint_path="${SAVE_PATH}" \
    --load_pretrain_checkpoint_path="${MODEL_PATH}/ernie.ckpt" \
    --train_data_file_path="${DATA_PATH}/${TASK_TYPE}/${TASK_TYPE}_train.mindrecord" \
    --eval_data_file_path="${DATA_PATH}/${TASK_TYPE}/${TASK_TYPE}_dev.mindrecord" \
    --schema_file_path="" > ${GLOG_log_dir}/train_${TASK_TYPE}_log.txt 2>&1 &