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
mkdir -p ms_log
mkdir -p save_models
CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models
DATA_PATH=${CUR_DIR}/data
SAVE_PATH=${CUR_DIR}/save_models
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_NUM=4
export RANK_TABLE_FILE=$PATH1

START_DEVICE_NUM=4
for((i=0; i<$DEVICE_NUM; i++))
do
    export DEVICE_ID=`expr $i + $START_DEVICE_NUM`
    export RANK_ID=$i
    if [ $i -eq 0 ]
    then
        DO_EVAL="true"
    else
        DO_EVAL="false"
    fi
    python ${CUR_DIR}/run_ernie_classifier.py \
        --task_type="xnli" \
        --device_target="Ascend" \
        --run_distribute="true" \
        --do_train="true" \
        --do_eval=$DO_EVAL \
        --device_num=$DEVICE_NUM \
        --device_id=$DEVICE_ID \
        --rank_id=$i \
        --epoch_num=3 \
        --num_labels=3 \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --train_batch_size=64 \
        --eval_batch_size=64 \
        --save_finetune_checkpoint_path="${SAVE_PATH}" \
        --load_pretrain_checkpoint_path="${MODEL_PATH}/ernie.ckpt" \
        --train_data_file_path="${DATA_PATH}/xnli/xnli_train.mindrecord0" \
        --eval_data_file_path="${DATA_PATH}/xnli/xnli_dev.mindrecord" > ${GLOG_log_dir}/train_xnli_log_$i.txt 2>&1 &
done