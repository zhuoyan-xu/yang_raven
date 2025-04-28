#!/bin/bash
#SBATCH --account=yguo258
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=evaluate-nq
#SBATCH --partition research

python subsample.py data/corpora/wiki/enwiki-dec2018/infobox.jsonl data/corpora/wiki/enwiki-dec2018/infobox_subsample001.jsonl 0.01
python subsample.py data/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec2018/text-list-100-sec_subsample001.jsonl 0.01
python subsample.py data/corpora/wiki/enwiki-dec2018/infobox.jsonl data/corpora/wiki/enwiki-dec2018/infobox_subsample01.jsonl 0.1
python subsample.py data/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec2018/text-list-100-sec_subsample01.jsonl 0.1
