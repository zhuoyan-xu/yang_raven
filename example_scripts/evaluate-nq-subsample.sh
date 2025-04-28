#!/bin/bash
#SBATCH --account=yguo258
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=evaluate-nq-subsample
#SBATCH --mem 60GB
#SBATCH --partition lianglab


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

export SHARE_OUTPUT=$(pwd)
export MASTER_ADDR=localhost

size=xl
DATA_DIR="${SHARE_OUTPUT}/data"
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
EXPERIMENT_NAME=$SLURM_JOB_ID-${name}-sub${SUBSAMPLE}
PRECISION="bf16"

# TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl"
# INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2021/infobox.jsonl"
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec_subsample${SUBSAMPLE}.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox_subsample${SUBSAMPLE}.jsonl"
PASSAGES="${TEXTS} ${INFOBOXES}"
CONTEXTS="${DATA_DIR}/nq_data/train.jsonl"

PRETRAINED_INDEX=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2018-sub${SUBSAMPLE}/passages
PRETRAINED_INDEX2=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2018-sub${SUBSAMPLE}/contexts

mkdir -p ${PRETRAINED_INDEX}
mkdir -p ${PRETRAINED_INDEX2}

CMD="python ${SHARE_OUTPUT}/evaluate.py \
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
    --load_index_path ${PRETRAINED_INDEX} \
    --load_index_path_data_retrieval ${PRETRAINED_INDEX2} \
    --passages ${PASSAGES} \
    --contexts ${CONTEXTS}
    "


srun -l \
    --output ${SHARE_OUTPUT}/logs/sub${SUBSAMPLE}/${EXPERIMENT_NAME}_$DATETIME.log ${CMD}
