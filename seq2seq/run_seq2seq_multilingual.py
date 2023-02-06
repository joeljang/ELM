# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from utils import modify_model_after_init, save_training_config
import shutil
import glob
from data import AutoPostProcessor
from third_party.models import T5Config, T5ForConditionalGeneration
from dataclasses import dataclass, field

from args_model import ModelArguments
from args_data import DataTrainingArguments
from args_training import TrainingArguments
from args_adapter import AdapterTrainingArguments

from third_party.trainers import Seq2SeqTrainer
from data import TaskDataCollatorForSeq2Seq
from data import AutoTask
from utils import get_adapter_config
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
import transformers
from datasets import load_dataset, load_metric, concatenate_datasets
from typing import Optional, List
import subprocess
import sys
import functools
import logging
from pytz import common_timezones
import torch
import os
import wandb

from data.tasks import TASK_MAPPING
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

logger = logging.getLogger(__name__)

def run_command(command):
    output = subprocess.getoutput(command)
    return output

TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                   "cola": ['matthews_correlation'],
                   "stsb": ['pearson', 'spearmanr'],
                   'sst2': ['accuracy'],
                   "mnli": ["accuracy"],
                   "mnli_mismatched": ["accuracy"],
                   "mnli_matched": ["accuracy"],
                   "qnli": ["accuracy"],
                   "rte": ["accuracy"],
                   "wnli": ["accuracy"],
                   "qqp": ["accuracy", "f1"],
                   "superglue_boolq": ["accuracy"],
                   "super_glue_rte": ["accuracy"],
                   "super_glue_cb": ["f1_multiclass", "accuracy"],
                   "super_glue_copa": ["accuracy"],
                   "super_glue_multirc": ["f1", "em"],
                   "super_glue_wic": ["accuracy"],
                   "super_glue_wsc.fixed": ["accuracy"],
                   "super_glue_record": ["f1", "em"],
                   "multi_nli": ["accuracy"],
                   "squad": ["em", "f1"],
                   "snli": ["accuracy"],
                   "nq": ["em", "f1"],
                   "hotpot_qa": ["em", "f1","rougeL"],
                   "searchqa": ["em", "f1"],
                   "newsqa": ["em", "f1"],
                   "trivia_qa": ["em", "f1"],
                   "imdb": ["accuracy"],
                   "winogrande": ["accuracy"],
                   "scitail": ["accuracy"],
                   "amazon_polarity": ["accuracy"],
                   "yelp_polarity": ["accuracy"],
                   "lama": ["accuracy","rougeL","bleu"],
                   "lambada": ["accuracy","rougeL","bleu"],
                   "cos_e": ["accuracy"],
                   "paws": ["accuracy"],
                   "hellaswag":["accuracy","rougeL","bleu"],
                   "story_cloze":["accuracy","rougeL","bleu"],
                   "anli_r1":["accuracy","rougeL","bleu"],
                   "anli_r2":["accuracy","rougeL","bleu"],
                   "anli_r3":["accuracy","rougeL","bleu"],
                   "winogrande":["accuracy","rougeL","bleu"],
                   "commonsense_qa": ["accuracy","rougeL","bleu"],
                   "dream": ["accuracy","rougeL","bleu"],
                   "quail": ["accuracy","rougeL","bleu"],
                   "quartz":["accuracy","rougeL","bleu"],
                   "social_i_qa":["accuracy","rougeL","bleu"],
                   "wiqa":["accuracy","rougeL","bleu"],
                   "cosmos_qa":["accuracy","rougeL","bleu"],
                   "qasc":["accuracy","rougeL","bleu"],
                   "quarel":["accuracy","rougeL","bleu"],
                   "sciq":["accuracy","rougeL","bleu"],
                   "wiki_hop":["accuracy","rougeL","bleu"],
                   "amazon_polarity":["accuracy","rougeL","bleu"],
                   "app_reviews":["accuracy","rougeL","bleu","spearmanr"],
                   "imdb":["accuracy","rougeL","bleu"],
                   "rotten_tomatoes":["accuracy","rougeL","bleu"],
                   "yelp_review_full":["accuracy","rougeL","bleu"],
                   "glue_qqp":["accuracy","rougeL","bleu"],
                   "glue_mrpc":["accuracy","rougeL","bleu"],
                   "ag_news": ["accuracy","rougeL","bleu"],
                   "dbpedia_14": ["accuracy","rougeL","bleu"],
                   "trec": ["accuracy","rougeL","bleu"],
                   "paws": ["accuracy","rougeL","bleu"],
                   "adversarial_qa":["em","f1","rougeL","bleu"],
                   "quoref":["em","f1","rougeL","bleu"],
                   "ropes":["em","f1"],
                   "duorc":["em","f1","rougeL","bleu"],
                   "wiki_qa":["accuracy","bleu","rougeL"],
                   "common_gen":["bleu","rougeL"],
                   "wiki_bio":["bleu","rougeL"],
                   "cnn_dailymail":["bleu","rougeL"],
                   "gigaword":["bleu","rougeL"],
                   "multi_news":["bleu","rougeL"],
                   "samsum":["rougeL"],
                   "xsum":["bleu","rouge"]
                   }

# run_seq2seq parameters.

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    
    #TODO : Fix manual
    training_args.predict_with_generate = True
    training_args.output_dir = training_args.output_dir.replace('expert_weights','task_vector_checkpoints')
    training_args.per_device_train_batch_size = 1
    # Initializae wandb
    #if training_args.wandb_log:
    wandb.init(entity="lklab_kaist", project="ROE_experiments_ICLR", name=training_args.output_dir)


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    adapter_args.train_task_adapters=False
    adapter_args.prefix_tuning=False
    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    config.attn_prefix_tuning = model_args.attn_prefix_tuning
    config.attn_method = model_args.attn_method
    config.ignore_target = model_args.ignore_target
    config.shared_attn = model_args.shared_attn
    config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.temperature = model_args.temperature
    config.fix_attention = model_args.fix_attention
    adapter_config = get_adapter_config(
        adapter_args, data_args, training_args, config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config
    )
    
    if model_args.load_prefix_embeddings is True:
        if model_args.prompt_embedding_path is None:
            for name, param in model.named_parameters():
                if "prefix_shared" in name or "prefix" in name:
                    shared_params = [param]
        else:
            shared_params = []
            for path in model_args.prompt_embedding_path:
                shared_param = torch.load(path)
                shared_params.append(shared_param)
            if model_args.target_prompt_embedding_path is not None:
                target_prompt_embedding = torch.load(
                    model_args.target_prompt_embedding_path)

        if model_args.attn_prefix_tuning is True:
            if training_args.do_train is True and model_args.shared_attn is False:
                # Load all of the source prompts
                model.store_prefix_weights(shared_params)
                # Initialize the target task prompt embedding using the first prompts
                model.update_prefix_weights_single(shared_params[0])
            elif training_args.do_train is True and model_args.shared_attn is True:
                # Load all of the source prompts
                model.store_prefix_weights(shared_params)
                # Initialize the target task prompt embeddings using the first prompts.
                model.update_prefix_weights_multi(
                    shared_params[0], num_target=config.num_target)
            else:
                # For inference
                # Load all of the source prompts
                model.store_prefix_weights(shared_params)
                # Load the trained target task prompt.
                model.update_prefix_weights_single(target_prompt_embedding)

        else:
            if model_args.target_prompt_embedding_path is None:
                model.update_prefix_weights(shared_params)
            else:
                model.update_prefix_weights(
                    shared_params, target_prompt_embedding)

    # Load linear attention
    if model_args.load_attention is True and model_args.attn_path is not None:
        model.update_attention_weights(torch.load(model_args.attn_path))

    # Load projection-based attentions 
    if model_args.load_attention is True and model_args.attn_path_sub is not None:
        model.update_attention_weights_sub(model_args.attn_path_sub)

    # Load layer norm weights & biases 
    if model_args.load_layer_norm is True and model_args.layer_norm_dir is not None:
        model.update_layer_norm_weights(model_args.layer_norm_dir)

    model.resize_token_embeddings(len(tokenizer))
    model = modify_model_after_init(
        model, training_args, adapter_args, adapter_config)

    # Loading the trained adapter code
    #TODO : Fix manual setting
    model_args.load_adapter_weights=False
    
    # if model_args.load_adapter_weights:
    #     adapter_params = {}
    #     #lst = os.listdir(model_args.adapter_dir)
    #     lst = os.listdir(os.path.join(training_args.output_dir,"adapter_params"))
    #     for path in lst:
    #         #full_path = model_args.adapter_dir + '/' + path
    #         full_path = os.path.join(training_args.output_dir,"adapter_params",path)
    #         params = torch.load(full_path)
    #         path_ = path.split('.')
    #         path = ".".join(path_[:-1])
    #         adapter_params[path] = params
    #     load_cnt=0
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #         #if name in adapter_params:
    #             load_cnt+=1
    #             param.data = adapter_params[name].cuda()
    #     print(f'load count: {load_cnt}')
    #     print(f'Finished loading {len(adapter_params)} number of adapter parameter files')

    data_args.dataset_name = data_args.task_name
    data_args.eval_dataset_name = data_args.eval_dataset_name
    data_args.test_dataset_name = data_args.test_dataset_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name)

    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]

        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    # Metric, we assume we have only one training task. => FIXED!
    eval_metrics_dict = {dataset_name+"*"+eval_prompt: AutoTask.get(dataset_name, dataset_config_name, prompt=eval_prompt).metric
                    for dataset_name, dataset_config_name, eval_prompt in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.eval_prompts)}
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(data_args.eval_dataset_name)
    print()
    print(data_args.eval_dataset_config_name)
    print()
    print(eval_metrics_dict)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # TODO: Fix manual
    #training_args.do_train = False
    
    if training_args.do_train:
        if data_args.train_files is not None:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           prompt=train_prompt,
                                           seed=data_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=train_file)
                for dataset_name, dataset_config_name, train_file, train_prompt
                in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_files, data_args.train_prompts)]
            for td in train_datasets:
                print('$$$$$$$$$$$$$$')
                print(len(td))
                print('$$$$$$$$$$$$$$')
        else:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           prompt=train_prompt,
                                           seed=data_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=data_args.train_file)
                for dataset_name, dataset_config_name, train_prompt
                in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_prompts)]
            print('***')
            print(len(train_datasets))
            print('***')
            for td in train_datasets:
                print('$$$$$$$$$$$$$$')
                print(len(td))
                print('$$$$$$$$$$$$$$')

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name,prompt=train_prompt).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length, )
            for dataset_name, dataset_config_name, train_prompt in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_prompts)]

        for i, train_dataset in enumerate(train_datasets):
            if model_args.shared_attn is True:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[i], task_id=i),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                print('$$$$$$$$$$$$$$')
                print(len(train_datasets[i]))
                print('$$$$$$$$$$$$$$')
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[i]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                print('$$$$$$$$$$$$$$')
                print(len(train_datasets[i]))
                print('$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$')
        print('TOTAL')
        print(len(train_dataset))
        print('$$$$$$$$$$$$$$')
        train_dataset = concatenate_datasets(train_datasets)
        print('$$$$$$$$$$$$$$')
        print('TOTAL')
        print(len(train_dataset))
        print('$$$$$$$$$$$$$$')
    # TODO: FIX MANUAL
    training_args.do_eval = False
    #data_args.max_val_samples=200
    
    if training_args.do_eval:
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name, prompt=eval_prompt).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name, eval_prompt in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.eval_prompts)]

        if data_args.validation_files is not None:
            eval_datasets = {eval_dataset+"*"+eval_prompt: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        prompt=eval_prompt,
                                                        seed=data_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=validation_file)
                for eval_dataset, eval_dataset_config, validation_file, eval_prompt in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.validation_files, data_args.eval_prompts)}
        else:
            eval_datasets = {eval_dataset+"*"+eval_prompt: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        prompt=eval_prompt,
                                                        seed=data_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
                for eval_dataset, eval_dataset_config, eval_prompt in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.eval_prompts)}
        
        for k, name in enumerate(eval_datasets):
                
            if name=="lama_fill_mask":
                max_target_lengths[k]=2
            elif name=="lambada_what comes next":
                max_target_lengths[k]=1
            if model_args.shared_attn is True:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    if training_args.do_test:
        if data_args.test_files is not None:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        prompt=test_prompt,
                                                        seed=data_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=test_file)
                for test_dataset, test_dataset_config, test_file, test_prompt in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_files, data_args.test_prompts)}
        else:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        prompt=test_prompt,
                                                        seed=data_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
                for test_dataset, test_dataset_config, test_prompt in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_prompts)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name, prompt=test_prompt).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name, test_prompt in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_prompts)]
        for k, name in enumerate(test_datasets):
            if name=="lama":
                max_target_lengths[k]=2
            elif name=="lambada":
                max_target_lengths[k]=1
            if model_args.shared_attn is True:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of
    # the data.
    if training_args.do_eval:
        data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]+"*"+data_args.eval_prompts[0]]['extra_fields'],
                    "test": test_datasets[data_args.test_dataset_name[0]+"*"+data_args.test_prompts[0]]['extra_fields'] if training_args.do_test else None,
                    "train": train_dataset['extra_fields'] if training_args.do_train else None}
    else:
        data_info = {"train": train_dataset['extra_fields'] if training_args.do_train else None}

    def compute_metrics(eval_preds, task_name):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(task_name, tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        result = {}
        eval_metrics = eval_metrics_dict[task_name]
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        with open(f"./output_logs/{data_args.dataset_name[0].replace('/',' ').replace('-',' ')}/{data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{task_name.replace('/',' ').replace('-',' ')}.txt",'a') as f:
            f.write("####################\n")
            f.write(task_name)
            f.write("\n")
            for a,b in zip(decoded_preds, decoded_labels):
                f.write(a)
                f.write("  ")
                f.write(b)
                f.write("\n")
            f.write(">> ")
            for key,value in result.items():
                f.write(str(key)+" : "+str(value)+" | ")
            f.write("\n")
            f.write("####################\n")
        return result

    if model_args.attn_learning_rate is not None:
        # Initialize a customized optimizer to set a different learning rate for the attention module.
        all_parameters = set(model.parameters())
        attn_params = []
        for name, param in model.named_parameters():
            if name == "encoder.attn_W_up" or name == "encoder.attn_W_down" or name == "encoder.layer_norm":
                attn_params += list(param)
        attn_params = set(attn_params)
        non_attn_params = all_parameters - attn_params
        non_attn_params = list(non_attn_params)
        attn_params = list(attn_params)

        optim = AdamW([
            {'params': non_attn_params},
            {'params': attn_params, 'lr': model_args.attn_learning_rate},
        ], lr=training_args.learning_rate,)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=len(
                train_dataset) * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
        )
        
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_args=data_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_datasets=eval_datasets if training_args.do_eval else None,
            #eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            #TODO : Fix manual
            #compute_metrics=compute_metrics,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            #evaluation_metrics=TASK_TO_METRICS[data_args.dataset_name[0]],
            #evaluation_metrics={dn : TASK_TO_METRICS[dn] for dn in data_args.dataset_name},
            evaluation_metrics = eval_metrics_dict,
            shared=model_args.shared_attn,
            optimizers=(optim, scheduler)
        )

    else:
        print('############################')
        print('FINAL TRAIN DATASET LEN')
        print(len(train_dataset))
        print('############################')
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_args=data_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_datasets=eval_datasets if training_args.do_eval else None,
            #eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            #TODO : Fix manual
            #compute_metrics=compute_metrics,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            #evaluation_metrics=TASK_TO_METRICS[data_args.dataset_name[0]],
            #evaluation_metrics={dn : TASK_TO_METRICS[dn] for dn in data_args.dataset_name},
            evaluation_metrics = eval_metrics_dict,
            shared=model_args.shared_attn)
    if training_args.do_eval:
        print('@@@@@@@@@@@@')
        print(eval_datasets)
        # {'paws': Dataset({
        #     features: ['attention_mask', 'extra_fields', 'input_ids', 'labels', 'task'],
        #     num_rows: 8000
        # }), 'superglue-rte': Dataset({
        #     features: ['attention_mask', 'extra_fields', 'input_ids', 'labels', 'task'],
        #     num_rows: 138
        # })}
        print('@@@@@@@@@@@@')
    # Saves training config.
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_training_config(sys.argv[1], training_args.output_dir)

    # TODO : FIX MANUAL
    model_args.save_adapter_weights = False
    params_to_save = {}
    unfrozen_layers = 0
    for name, param in trainer.model.named_parameters():
        if param.requires_grad == True:
            print(name)
            params_to_save[name] = 0
            unfrozen_layers+=1
    print(f'number of unfrozen layers (for beginning of training model): {unfrozen_layers}')

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})
        param_dict = {}
        for name, param in trainer.model.named_parameters():
            param_dict[name]=param.clone().detach().cpu()
        torch.save(param_dict, training_args.output_dir+'_last.pt')

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        train_metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        if not model_args.save_prefix_only:
            trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    if model_args.shared_attn is True and model_args.ignore_target is False:
        learned_embeddings = trainer.model.encoder.prefix_emb.clone().detach()
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if model_args.shared_attn is True:
            #for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
            for task, eval_dataset in eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
                                           task = task)
                trainer.log_metrics(f"eval_{task}_", metrics)
                trainer.save_metrics(f"eval_{task.replace('/',' ')}_", metrics)
                if training_args.wandb_log:
                    wandb.log({f"eval_{task}_": metrics})
                
        else:
            for task, eval_dataset in eval_datasets.items():
                print('$$$$$$$$$$$$@@@@@@@@@@@@@@')
                print(task)
                print('$$$$$$$$$$$$@@@@@@@@@@@@@@')
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
                                           task = task)
                trainer.log_metrics(f"eval_{task}_", metrics)
                trainer.save_metrics(f"eval_{task.replace('/',' ')}_", metrics)
                if training_args.wandb_log:
                    wandb.log({f"eval_{task}_": metrics})

    # remove checkpoint for the save_prefix_only setting to avoid overly saving models.
    if model_args.save_prefix_only:
        checkpoints = glob.glob(os.path.join(
            training_args.output_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoints:
            # save models
            if not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                continue
            checkpoint_model = torch.load(os.path.join(
                os.path.join(checkpoint_dir, "pytorch_model.bin")))
            new_dir = "{}_prompt_only".format(checkpoint_dir)
            os.mkdir(new_dir)
            for name, param in checkpoint_model.items():
                if model_args.attn_prefix_tuning is False and ("prefix_shared" in name or "prefix" in name):
                    shared_params = param
                    torch.save(shared_params, os.path.join(
                        training_args.output_dir, "prefix_embeddings.pt"))
                elif model_args.attn_prefix_tuning is True and name == "prefix_shared":
                    shared_params = param
                    if model_args.shared_attn is True:
                        for i in range(config.num_target):
                            torch.save(shared_params[i], os.path.join(
                                new_dir, "prefix_embeddings_{}.pt".format(i)))
                    else:
                        torch.save(shared_params, os.path.join(
                            new_dir, "prefix_embeddings.pt"))
                if model_args.attn_prefix_tuning is True and "encoder.attn_Wa.weight" == name:
                    attn_weights_params = param
                    torch.save(attn_weights_params, os.path.join(
                        new_dir, "attn_Wa_weights.pt"))
                if model_args.attn_prefix_tuning is True and "encoder.attn_W_down.weight" == name:
                    attn_weights_params = param
                    torch.save(attn_weights_params, os.path.join(
                        new_dir, "attn_W_down.pt"))
                if model_args.attn_prefix_tuning is True and "encoder.attn_W_up.weight" == name:
                    attn_weights_params = param
                    torch.save(attn_weights_params, os.path.join(
                        new_dir, "attn_W_up.pt"))
                if model_args.attn_prefix_tuning is True and "encoder.layer_norm.weight" == name:
                    attn_weights_params = param
                    torch.save(attn_weights_params, os.path.join(
                        new_dir, "layer_norm_weight.pt"))
                if model_args.attn_prefix_tuning is True and "encoder.layer_norm.bias" == name:
                    attn_weights_params = param
                    torch.save(attn_weights_params, os.path.join(
                        new_dir, "layer_norm_bias.pt"))
            # after saving prompts, we will remove unnecessary checkpoint dir.
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print("Error: %s : %s" % (checkpoint_dir, e.strerror))

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        if model_args.shared_attn is True:
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                trainer.model.encoder.prefix_emb[0].data = learned_embeddings[idx]
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
                                           metric_key_prefix="test", task= task
                                           )
                trainer.log_metrics(f"test_{task}_", metrics)
                trainer.save_metrics(f"test_{task.replace('/',' ')}_", metrics)

        else:
            for task, test_dataset in test_datasets.items():
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
                                           metric_key_prefix="test", task= task
                                           )
                trainer.log_metrics(f"test_{task}_", metrics)
                trainer.save_metrics(f"test_{task.replace('/',' ')}_", metrics)

    return results

def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()