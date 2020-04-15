#!/bin/bash

# The following may be modified
# Model can be "bert_large" or "bert_base"
MODEL="bert_large"
# Sequence length can be 128, 256, 512, or other number of choice
SEQ=128
# Batch size can be anything that fits the GPU
BATCH=8

# Print a message
echo "This script will run the $MODEL model."
echo "Below is what it will do:"
echo "  1. Prepare a sample traing data set in a temporary directory;"
echo "  2. Train $MODEL;"
echo "  3. Clean up the temoprary directory created and exit."
echo "Please press any key to start, or ESC to exit."

# Read a key press
read -n 1 -s KEY
if [ "$KEY" == $'\e' ] ; then
  echo "Exit. Did nothing."
  exit 0
fi

# Get the folders
SCRIPTPATH=$(dirname $(realpath $0))
CODE_DIR=$SCRIPTPATH/..
CODE_DIR_INSIDE=.

# Preparing temporary folder for training
TRAIN_DIR_NAME=dashboard_train_dir
TRAIN_DIR=$CODE_DIR/$TRAIN_DIR_NAME
rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR
TRAIN_DIR_INSIDE=$CODE_DIR_INSIDE/$TRAIN_DIR_NAME

# Copy the configuration file and vocab file
MODEL_CONFIG_DIR=$CODE_DIR/configs/$MODEL
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# Preparing training data
echo
echo "=== Preparing training data ==="

# Generate the training data
DATA_SOURCE_FILE_PATH=sample_text.txt
DATA_SOURCE_NAME=$(basename "$DATA_SOURCE_FILE_PATH")

DATA_TFRECORD=${DATA_SOURCE_NAME}_seq${SEQ}.tfrecord

# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
  echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

if [ ! -f "$DATA_TFRECORD" ]; then
  # generate tfrecord of data
  python3 $CODE_DIR_INSIDE/create_pretraining_data.py \
    --input_file=$CODE_DIR_INSIDE/$DATA_SOURCE_FILE_PATH \
    --output_file=$TRAIN_DIR_INSIDE/$DATA_TFRECORD \
    --vocab_file=$TRAIN_DIR_INSIDE/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
    --masked_lm_prob=$MASKED_LM_PROB \
    --random_seed=12345 \
    --dupe_factor=5
fi

# Perform training
echo
echo "=== Training BERT ==="
CUR_TRAINING=${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}
mkdir -p $TRAIN_DIR/$CUR_TRAINING

TRAIN_STEPS=200
WARMUP_STEPS=50
LEARNING_RATE=2e-5

# export HIP_VISIBLE_DEVICES=0 # choose gpu
# run pretraining
python3 $CODE_DIR_INSIDE/run_pretraining.py \
  --input_file=$TRAIN_DIR_INSIDE/$DATA_TFRECORD \
  --output_dir=$TRAIN_DIR_INSIDE/$CUR_TRAINING \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$TRAIN_DIR_INSIDE/bert_config.json \
  --train_batch_size=$BATCH \
  --max_seq_length=$SEQ \
  --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
  --num_train_steps=$TRAIN_STEPS \
  --num_warmup_steps=$WARMUP_STEPS \
  --learning_rate=$LEARNING_RATE \
  2>&1 | tee $TRAIN_DIR/$CUR_TRAINING/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}.txt

# Cleaning up
echo
echo "=== Cleaning up ==="
echo "Delete folder holding training data"
rm -rf $TRAIN_DIR_INSIDE
