if [ $# -ne 2 ]
then
    echo "Usage: sh run_distribute_finetune.sh [RANK_TABLE_FILE] [DATASET_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

DATASET_PATH=$(get_real_path $2)
echo $DATASET_PATH

if [ ! -f $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    python ${CUR_DIR}/run_ernie_classifier.py  \
        --device_target="Ascend" \
        --do_train="true" \
        --do_eval="true" \
        --device_id=$DEVICE_ID \
        --epoch_num=10 \
        --num_class=2 \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --train_batch_size=24 \
        --eval_batch_size=32 \
        --save_finetune_checkpoint_path="${SAVE_PATH}" \
        --load_pretrain_checkpoint_path="${MODEL_PATH}/ernie.ckpt" \
        --train_data_file_path="${DATA_PATH}/chnsenticorp/chnsenticorp_train.mindrecord" \
        --eval_data_file_path="${DATA_PATH}/chnsenticorp/chnsenticorp_dev.mindrecord" \
        --schema_file_path="" > ${GLOG_log_dir}/train_classifier_log_$i.txt 2>&1 &
done