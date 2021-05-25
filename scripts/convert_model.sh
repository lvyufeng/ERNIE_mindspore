PROJECT_DIR=`pwd`
MODEL_PATH=${PROJECT_DIR}/pretrain_models/ernie
CONVERT_PATH=${PROJECT_DIR}/pretrain_models/converted
python ${PROJECT_DIR}/src/convert.py  \
    --input_dir="${MODEL_PATH}" \
    --output_dir="${CONVERT_PATH}"