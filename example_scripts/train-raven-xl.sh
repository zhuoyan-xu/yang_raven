#!/bin/bash
#SBATCH --account=yguo258
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=train-raven-xl
#SBATCH --mem 60GB
#SBATCH --partition lianglab


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

export SHARE_OUTPUT=$(pwd)
export MASTER_ADDR=localhost

size=xl
DATA_DIR="${SHARE_OUTPUT}/data"
SUBSAMPLE=01

# Prepare train/dev/test data from corpus:
# TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl"
# INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"

shuf ${TEXTS} > "${TEXTS}.shuf"
head -n 2000 "${TEXTS}.shuf" | head -n 1000 > "${TEXTS}.shuf.test"
head -n 2000 "${TEXTS}.shuf" | tail -n 1000 > "${TEXTS}.shuf.valid"
tail -n +2000 "${TEXTS}.shuf" > "${TEXTS}.shuf.train"

shuf ${INFOBOXES} > "${INFOBOXES}.shuf"
head -n 2000 "${INFOBOXES}.shuf" | head -n 1000 > "${INFOBOXES}.shuf.test"
head -n 2000 "${INFOBOXES}.shuf" | tail -n 1000 > "${INFOBOXES}.shuf.valid"
tail -n +2000 "${INFOBOXES}.shuf" > "${INFOBOXES}.shuf.train"

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILES="${TEXTS}.shuf.train ${INFOBOXES}.shuf.train"
EVAL_FILES="${TEXTS}.shuf.valid ${INFOBOXES}.shuf.valid"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=RAVEN-${size}-train
PRECISION="bf16"

PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
# PRETRAINED_MODEL=none  # need to set this as none if it is not the initial run
PRETRAINED_INDEX=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org-enwiki-dec2018-sub${SUBSAMPLE}/passages

CMD="python train.py \
    --model "raven" \
    --use_gradient_checkpoint_reader \
    --shard_grads --shard_optim \
    --precision ${PRECISION} \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --passages ${TRAIN_FILES} \
    --dropout 0.1 \
    --lr 4e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 384 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --save_freq 1000 --eval_freq 500 --log_freq 50 \
    --total_steps 5000 \
    --warmup_steps 100 \
    --main_port $port \
    --min_words_per_lm_instance 10 \
    --task "mlm" \
    --load_index_path ${PRETRAINED_INDEX} \
    --save_index_n_shards 128"
# save index n shards needs to be divided by the number of GPUs

srun -l \
    --output ${SHARE_OUTPUT}/logs/sub${SUBSAMPLE}/${EXPERIMENT_NAME}_$DATETIME.log ${CMD}

