#!/bin/bash


echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Run name: $1"
echo "Use wandb: $2"


# Print GPU information
echo "GPU Information:"
nvidia-smi

# export HF_TOKEN="<YOUR HF TOKEN>"

# export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
# export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
# export TRITON_CACHE_DIR=$_CONDOR_SCRATCH_DIR/.triton
export HF_HOME=/staging/zxu444/yang_raven/.cache/huggingface

# huggingface-cli login --token $HF_TOKEN

echo "================ start running ================"

# Set variables (keeping the same values from your SLURM script)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
export SHARE_OUTPUT=$(pwd)
export MASTER_ADDR=localhost
size=xl
# DATA_DIR="${SHARE_OUTPUT}/data"
DATA_DIR="/staging/zxu444/yang_raven/data"
SUBSAMPLE=001
port=$(shuf -i 15000-16000 -n 1)
n_shot=5
fusion=0
doc=40
model=atlas
# model=raven
name="nq-${model}-${size}-${n_shot}-shot-${fusion}-fusion-d${doc}-sub${SUBSAMPLE}"
# data='triviaqa_data'
data='nq_data'
TRAIN_FILE="${DATA_DIR}/${data}/train.jsonl"
EVAL_FILES="${DATA_DIR}/${data}/test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/${model}/${size}
SAVE_DIR=${DATA_DIR}/experiments/sub${SUBSAMPLE}/
EXPERIMENT_NAME=$(date +%s)-${name}-sub${SUBSAMPLE}  # Using timestamp instead of SLURM job ID
PRECISION="bf16"
# TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl"
# INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2021/infobox.jsonl"
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2021/text-list-100-sec_subsample${SUBSAMPLE}.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2021/infobox_subsample${SUBSAMPLE}.jsonl"
PASSAGES="${TEXTS} ${INFOBOXES}"
CONTEXTS="${DATA_DIR}/nq_data/train.jsonl"
PRETRAINED_INDEX=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2021-sub${SUBSAMPLE}/passages
PRETRAINED_INDEX2=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2021-sub${SUBSAMPLE}/contexts

# Create directories if they don't exist
mkdir -p ${PRETRAINED_INDEX}
mkdir -p ${PRETRAINED_INDEX2}
mkdir -p ${SHARE_OUTPUT}/logs/sub${SUBSAMPLE}

# Run the command directly instead of using srun
python ${SHARE_OUTPUT}/evaluate_TTT.py \
 --model ${model} \
 --n_shots ${n_shot} \
 --fusion ${fusion} \
 --gold_score_mode "pdist" \
 --precision ${PRECISION} \
 --text_maxlength 512 \
 --target_maxlength 512 \
 --reader_model_type google/t5-${size}-lm-adapt \
 --model_path ${PRETRAINED_MODEL} \
 --train_data ${TRAIN_FILE} \
 --eval_data ${EVAL_FILES} \
 --per_gpu_batch_size 1 \
 --n_context ${doc} --retriever_n_context ${doc} \
 --name ${EXPERIMENT_NAME} \
 --checkpoint_dir ${SAVE_DIR} \
 --main_port $port \
 --write_results \
 --task qa \
 --index_mode flat \
 --passages ${PASSAGES} \
 --contexts ${CONTEXTS} \
 > ${SHARE_OUTPUT}/logs/sub${SUBSAMPLE}/${EXPERIMENT_NAME}_$DATETIME.log 2>&1

## --checkpoint_dir ${SAVE_DIR} --> is in staging
## ${SHARE_OUTPUT}/logs --> is transferred out by chtc job

#  --load_index_path ${PRETRAINED_INDEX} \
#  --load_index_path_data_retrieval ${PRETRAINED_INDEX2} \