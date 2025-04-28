#!/bin/bash
#SBATCH --account=yguo258
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=evaluate-nq
#SBATCH --mem=0
#SBATCH --partition lianglab

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

export SHARE_OUTPUT=$(pwd)
export MASTER_ADDR=localhost

size=xl
DATA_DIR="${SHARE_OUTPUT}/data"

port=$(shuf -i 15000-16000 -n 1)

n_shot=5
fusion=0
doc=10

model=atlas
# model=raven
name="nq-${model}-${size}-${n_shot}-shot-${fusion}-fusion-d${doc}"

# data='triviaqa_data'
data='nq_data'
TRAIN_FILE="${DATA_DIR}/${data}/train.jsonl"
EVAL_FILES="${DATA_DIR}/${data}/test.jsonl"

PRETRAINED_MODEL=${DATA_DIR}/models/${model}/${size}

SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${name}
PRECISION="bf16"

# TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl"
# INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2021/infobox.jsonl"
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"
PASSAGES="${TEXTS} ${INFOBOXES}"

PRETRAINED_INDEX=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2018

CMD="python ${SHARE_OUTPUT}/evaluate2.py \
    --model ${model} \
    --n_shots ${n_shot} \
    --fusion ${fusion} \
    --gold_score_mode "pdist" \
    --precision ${PRECISION} \
    --text_maxlength 256 \
    --target_maxlength 256 \
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
    --save_index_path ${PRETRAINED_INDEX} \
    --passages ${PASSAGES} \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever"

echo "${SHARE_OUTPUT}/logs/${EXPERIMENT_NAME}_$DATETIME.log"
srun -l \
    --output ${SHARE_OUTPUT}/logs/${EXPERIMENT_NAME}_$DATETIME.log ${CMD}
