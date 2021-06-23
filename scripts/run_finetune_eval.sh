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
    echo "sh run_finetune_eval.sh TASK_TYPE"
    echo "for example: sh run_finetune_eval.sh msra_ner"
    echo "TASK_TYPE including [msra_ner, chnsenticorp]"
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
DEVICE_ID=5
case $TASK_TYPE in
  "msra_ner")
    python ${CUR_DIR}/run_ernie_ner.py  \
        --device_target="Ascend" \
        --number_labels=7 \
        --label_map_config="${DATA_PATH}/msra_ner/label_map.json" \
        --do_train="false" \
        --do_eval="true" \
        --device_id=$DEVICE_ID \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --eval_batch_size=32 \
        --load_finetune_checkpoint_path="${SAVE_PATH}/ner-6_1304.ckpt" \
        --eval_data_file_path="${DATA_PATH}/msra_ner/msra_ner_test.mindrecord" \
        --schema_file_path="" > ${GLOG_log_dir}/eval_ner_log.txt 2>&1 &
    ;;
  "chnsenticorp")
    python ${CUR_DIR}/run_ernie_classifier.py  \
        --device_target="Ascend" \
        --do_train="false" \
        --do_eval="true" \
        --device_id=$DEVICE_ID \
        --num_class=2 \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --eval_batch_size=32 \
        --load_finetune_checkpoint_path="${SAVE_PATH}/classifier-10_400.ckpt" \
        --eval_data_file_path="${DATA_PATH}/chnsenticorp/chnsenticorp_test.mindrecord" \
        --schema_file_path="" > ${GLOG_log_dir}/eval_classifier_log.txt 2>&1 &
    ;;
  esac