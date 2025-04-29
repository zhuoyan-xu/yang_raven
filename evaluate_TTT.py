# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict

# import GPUtil
# from tabulate import tabulate

# def check_gpu(): 
#     gpus = GPUtil.getGPUs()
#     gpu_list = [[gpu.id, gpu.name, f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB", f"{gpu.load * 100:.1f}%"] for gpu in gpus]
#     print(tabulate(gpu_list, headers=["ID", "GPU", "Memory Usage", "GPU Load"]))


import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src.index import DistributedFAISSIndex, DistributedIndex
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
import random

import copy
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator

def evaluate_TTT(model, index, data_path, index2, optimizer, scheduler, step, opt, checkpoint_path):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [[""]])
        target = batch.get("target", [[""]])
        batch_metadata = batch.get("metadata")
        
        query_retriever = [x[-1] for x in query]

        if opt.load_index_path_data_retrieval: # add in-context examples later
            query =  [x[-1] for x in query]
        else:
            query = [" ".join(x) for x in query]
        
        pre_target = ["" for x in target] 
        target = [x[-1] for x in target]
        
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize_multi_chunk(query_retriever, pre_target, target)

        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _ = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
            if opt.load_index_path_data_retrieval and opt.n_shots>0:
                retrieved_examples, _ = unwrapped_model.retrieve(
                    index2,
                    opt.n_shots,
                    query,
                    query_ids_retriever,
                    query_mask_retriever,
                    batch_metadata=batch_metadata,
                    filtering_fun=None,
                )
                
                 # otherwise, query will be updated later
                if (len(query) == 0) or (len(query[0]) == 0) or query[0][0]=="":
                    continue
                
                query_ = []
                for j,q in enumerate(query):
                    re_examples = []
                    for re_exp in retrieved_examples[j]:
                        if "target" in re_exp:
                            re_tgt = re_exp["target"]
                        elif "answers" in re_exp:
                            re_tgt = random.choice(re_exp["answers"])
                        elif "answer" in re_exp:
                            re_tgt = re_exp["answer"]
                        re_examples.append("Question: " + re_exp['question'] + " Answer: " + re_tgt + " ")
                    q = re_examples + [q]
                    query_.append(q)
                query = query_
                query = [" ".join(x) for x in query]
                
        else:
            if opt.closed_book:
                # opt.encoder_format = "{query}"
                opt.retriever_format = ""
                retrieved_passages = [[{}]]*len(query)
            else:
                assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
                retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

        TTT_model = copy.deepcopy(unwrapped_model)
        tune_model(TTT_model, query_retriever, retrieved_examples,
                 index, optimizer, scheduler, opt, checkpoint_path)
        
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0) or query[0][0]=="":
            continue
        
        reader_tokens, _ = TTT_model.tokenize_passages(query, retrieved_passages)

        if "eval_loss" in task.metrics:
            eval_loss, logits = TTT_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        generation = TTT_model.generate(
            reader_tokens, pre_target, choices=batch["choices"] if "choices" in batch else None
        )

        del TTT_model
        torch.cuda.empty_cache()

        for k, g in enumerate(generation):
                
            query_ids = reader_tokenizer.encode(
                pre_target[k], add_special_tokens=False
            )
            
            g = g[len(query_ids) + 1 :]    
            
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            pred = pred.split("Question:")[0]
            pred = pred.split("Claim:")[0]
            
            gold = [target[k]] if not "answers" in batch else batch["answers"][k]
            
            sample_metrics = task.evaluation(pred, gold)
            for key, value in sample_metrics.items():
                metrics[key].append(value)

            if opt.write_results:
                ex = {"query": query[k], "answers": gold, "generation": pred}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics

def train_TTT(
    model,
    index,
    optimizer,
    scheduler,
    opt,
    checkpoint_path,
    queries,
    targets,
    epochs: int = 5,
    lr: float = 1e-4
):
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)

    scale = 2.0
    grad_stats = defaultdict(lambda: [])
    task = get_task(opt, unwrapped_model.reader_tokenizer)

    step = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("lr:", lr)

    while step < epochs:
        model.train()

        step += 1
        
        reader_loss, _ = model(
            index=index,
            query=queries,
            target=targets,
            filtering_fun=task.filter,
        )

        train_loss = reader_loss

        backward_start = time.time()

        train_loss = scale * train_loss
        train_loss.backward()

        model_update_start = time.time()
        stats = util.compute_grad_stats(model)
        if stats["skip_example"]:
            model.zero_grad()
            # continue
        else:
            for k, v in stats.items():
                grad_stats[k].append(v)

        if len(grad_stats["max"]) >= THRESHOLD_GRAD_STATS:
            if np.mean(grad_stats["max"]) > GRAD_SCALE_UPPER_BOUND_MEAN:
                scale /= 2
            elif np.mean(grad_stats["mean"]) < GRAD_SCALE_LOWER_BOUND_MEAN:
                scale *= 2
            grad_stats.clear()

        if step % opt.accumulation_steps == 0 and not stats["skip_example"]:
            if opt.is_distributed and opt.shard_optim:
                optimizer.clip_grad_norm(scale * opt.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), scale * opt.clip)

            optimizer.step(scale=scale)
            model.zero_grad()

        if step > opt.total_steps:
            exit()

def freeze_layers(model):
    for param in model.retriever.parameters():
        param.requires_grad = False

    for param in model.reader.parameters():
        param.requires_grad = False

    for param in model.reader.lm_head.parameters():
        param.requires_grad = True


def tune_model(model, query, retrieved_examples, index, optimizer, scheduler, opt, checkpoint_path): 
    start = time.time()

    # Fix the weights of the early layers of the model
    freeze_layers(model)

    print("query: ", query)
    print("retrieved_examples: ", retrieved_examples)

    # TTT: No retrieval version. [Design choice. Do we reshuffle x1, x2, ..., xn]
    # [x2, y2, ..., xn, yn, x1]  --> y1
    # ...
    # [x1, y1, ..., xn-1, yn-1, xn]  --> yn
    # n total training examples 
    for i, _ in enumerate(query):
        train_queries = []
        train_targets = []
        for j in range(len(retrieved_examples[i])):
            re_examples = []
            q = ["Question: " + retrieved_examples[i][j]['question'] + " Answer:<extra_id_0>"]
            for k, re_exp in enumerate(retrieved_examples[i]):
                if k == j:
                    continue

                if "target" in re_exp:
                    re_tgt = re_exp["target"]
                elif "answers" in re_exp:
                    re_tgt = random.choice(re_exp["answers"])
                elif "answer" in re_exp:
                    re_tgt = re_exp["answer"]
                re_examples.append("Question: " + re_exp['question'] + " Answer: " + re_tgt + " ")
        
            q = re_examples + q
            q = [" ".join(q)]
            train_queries.append(q)
        
            re_exp = retrieved_examples[i][j]
            if "target" in re_exp:
                re_tgt = re_exp["target"]
            elif "answers" in re_exp:
                re_tgt = random.choice(re_exp["answers"])
            elif "answer" in re_exp:
                re_tgt = re_exp["answer"]
            train_targets.append([re_tgt])

        print("train_queries: ", train_queries)
        print("train_targets: ", train_targets)
        train_TTT(
            model, 
            index,
            optimizer,
            scheduler,
            opt,
            checkpoint_path,
            train_queries,
            train_targets
        )
        

    model.eval()
    end = time.time()
    print("Per TTT time: ", end - start)

    # Original version (prompt): [x1, y1, ..., xn, yn, retrieved_passages(xq), xq]
    # [x1, ..., xn] are retrieved examples of xq. 

    # TTT: Retrieval version. 
    # [x2, y2, ..., xn, yn, retrieved_passages(x1), x1]  --> y1
    # ...
    # [x1, y1, ..., xn-1, yn-1, retrieved_passages(xn), xn]  --> yn
    # n total training examples 

    # 1. naive retrieval 
    # 2. retrieval with cached index.

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")
   
    index, passages = load_or_initialize_index(opt)
    model, optimizer, scheduler, _, _, opt, step = load_or_initialize_atlas_model(opt)

    index2 = None
    if opt.load_index_path_data_retrieval is not None:
        if opt.index_mode == "flat":
            index2 = DistributedIndex()
        elif opt.index_mode == "faiss":
            index2 = DistributedFAISSIndex(opt.faiss_index_type, opt.faiss_code_size)
        else:
            raise ValueError(f"unsupported index mode {opt.index_mode}")

        logger.info(f"Loading index from: {opt.load_index_path_data_retrieval} with index mode: {opt.index_mode}")
        if opt.index_mode == "faiss":
            logger.info(f"loading faiss index type {opt.faiss_index_type} with parameters {opt.faiss_code_size}")
        index2.load_index(opt.load_index_path_data_retrieval, opt.save_index_n_shards)
        passages2 = [index2.doc_map[i] for i in range(len(index2.doc_map))]
    
    logger.info("Start Evaluation")
    dist_utils.barrier()

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)
    
    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if not opt.retrieve_only:
            metrics = evaluate_TTT(model, index, data_path, index2, optimizer, scheduler, step, opt, checkpoint_path)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)
