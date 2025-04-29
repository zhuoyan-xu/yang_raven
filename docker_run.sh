docker run --gpus all --user $(id -u):$(id -g) --rm=true -it \
  -e TRANSFORMERS_CACHE=/scratch/.cache \
  -e HF_HOME=/scratch/.cache/huggingface \
  -v $(pwd):/scratch -w /scratch zhuoyanxu/raven:v1.0 /bin/bash