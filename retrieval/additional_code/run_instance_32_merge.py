import json
import pprint 
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import torch
from promptsource.promptsource.templates import DatasetTemplates
import numpy as np
import pandas as pd
import operator
import argparse
from argparse import ArgumentParser
import re
from rank_bm25 import BM25Okapi
from math import sqrt, pow, exp

def euclidean_distance(x,y):
  """ return euclidean distance between two lists """
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

# Loading configs
parser = ArgumentParser()
parser.add_argument('--config', default=None, type=str)
arg_ = parser.parse_args()
if arg_.config is None:
    raise NameError("Include a config file in the argument please.")

# Getting configurations
config_path = arg_.config
with open(config_path) as config_file:
    config = json.load(config_file)
config = argparse.Namespace(**config)

if config.method=='ST':
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
elif config.method=='ST_MP':
    sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
elif config.method=='T0':
    state_dict = torch.load("rospr/T0_small.pt")
    t0_retriever = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
    t0_retriever = t0_retriever._load_state_dict_into_model(t0_retriever, state_dict, "rospr/T0_small.pt")[0]
    t0_retriever.to('cuda')
    t0_retriever = t0_retriever.get_encoder()
    tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")
elif config.method=='BM25':
    print('Doing BM25 for retrieval!')
else:
    raise Exception(f'{config.method} not yet implemented.')


print(f'Using {config.text_format} as query-key format')

if 'top_k' not in config:
    raise Exception('Provide config with top_k.')

def clease_jinja(jinja):
    jinja = re.sub(r'{{.+?}}', '', jinja)
    jinja = re.sub(r'{.+?}', '', jinja)
    jinja = jinja.replace("|||", "")
    jinja = jinja.replace("\n", "")
    return jinja

# Setting the text format to use for retrieval (task level)
def get_text_format(instance, answer_choice_format, label_list):
    if label_list==None:
        label_list = ''
    else:
        label_list = ", ".join(label_list)
    if config.text_format=='instance':
        text = f"{instance}"
    elif config.text_format=='label':
        text = f"Answer Choices: {label_list}"
    elif config.text_format=='choice':
        text = f"Answer Choices: {answer_choice_format}"
    elif config.text_format=='label_instance':
        text = f"Answer Choices: {label_list}, Instance: {instance}"
    elif config.text_format=='choice_instance':
        text = f"Answer Choices: {answer_choice_format}, Instance: {instance}"
    else:
        raise Exception('Select the correct TEXT_FORMAT')
    return text
    
def format_dataset_name(dataset_name):
    if dataset_name=='cos_e':
        return "cos_e/v1.11"
    elif dataset_name=='wiki_hop':
        return "wiki_hop/original"
    elif dataset_name=='paws':
        return "paws/labeled_final"
    elif dataset_name=='glue_qqp':
        return "glue/qqp"
    elif dataset_name=='glue_mrpc':
        return "glue/mrpc"
    elif dataset_name=='adversarial_qa':
        return "adversarial_qa/adversarialQA"
    elif dataset_name=='duorc':
        return "duorc/ParaphraseRC"
    elif dataset_name=='hotpot_qa':
        return "hotpot_qa/fullwiki"
    elif dataset_name=='cnn_dailymail':
        return "cnn_dailymail/3.0.0"
    elif dataset_name=='story_cloze':
        return "story_cloze/2016"
    elif dataset_name=='anli_r1' or dataset_name=='anli_r2' or dataset_name=='anli_r3':
        return 'anli'
    elif dataset_name=='super_glue_copa':
        return 'super_glue/copa'
    elif dataset_name=='super_glue_cb':
        return 'super_glue/cb'
    elif dataset_name=='super_glue_rte':
        return 'super_glue/rte'
    elif dataset_name=='super_glue_wsc.fixed':
        return 'super_glue/wsc.fixed'
    elif dataset_name=='super_glue_wic':
        return 'super_glue/wic'
    elif dataset_name=='winogrande':
        return 'winogrande/winogrande_xl'
    else:
        return dataset_name

def revert_dataset_name(dataset_name):
    if dataset_name=="cos_e/v1.11":
        return 'cos_e'
    elif dataset_name=="wiki_hop/original":
        return 'wiki_hop'
    elif dataset_name=="paws/labeled_final":
        return 'paws'
    elif dataset_name=="glue/qqp":
        return 'glue_qqp'
    elif dataset_name=="glue/mrpc":
        return 'glue_mrpc'
    elif dataset_name=="adversarial_qa/adversarialQA":
        return 'adversarial_qa'
    elif dataset_name=="duorc/ParaphraseRC":
        return 'duorc'
    elif dataset_name=="hotpot_qa/fullwiki":
        return 'hotpot_qa'
    elif dataset_name=="cnn_dailymail/3.0.0":
        return 'cnn_dailymail'
    elif dataset_name=='story_cloze/2016':
        return "story_cloze"
    else:
        return dataset_name

train_dict = json.load(open("train_examples_100.json","r"))
eval_dict = json.load(open("eval_examples_300.json","r"))

if 'ST' in config.method or 'T0' in config.method:
    all_embeddings = np.load(f"indexes/train_100_{config.method}_{config.text_format}.npy")


# Loading Each Expert Results for MoE Simulation 
score_list = json.load(open("score_list.json", "r"))
print(f'Loading the score list..!')
import random

score = {}
expert_cnt = 0
for expert_dataset in score_list:
    expert_prompt_lst = score_list[expert_dataset]
    for expert_prompt in expert_prompt_lst:
        #print(f'expert_dataset:{expert_dataset}, expert_prompt:{expert_prompt}')
        eval_samples_cnt=0
        eval_dataset_lst = expert_prompt_lst[expert_prompt]
        for eval_dataset in eval_dataset_lst:
            eval_prompt_lst = eval_dataset_lst[eval_dataset]
            for eval_prompt in eval_prompt_lst:
                samples = eval_prompt_lst[eval_prompt]
                for id in samples:
                    val = samples[id]
                    if '#' in val:
                        val = val.split(' ')[0]
                    val = float(val)
                    score[f'{expert_dataset}@{expert_prompt}@{eval_dataset}@{eval_prompt}@{id}'] = val
        expert_cnt+=1

# Loading Training Instances
train_index = {}
text_to_expert = {}
train_cnt = 0
corpus = []
print(f'Getting the training index ready..!')
for key in train_dict:
    dataset_name = format_dataset_name(key)
    prompts = train_dict[key]
    for prompt_name in prompts:
        samples = prompts[prompt_name]
        print(f'{dataset_name}, {prompt_name}, {len(samples)}')
        for row_index in samples:
            row = samples[row_index]
            prompt_name = row["prompt"]
            text = row["source"]
            target = row["target"]
            
            try:
                label_list = row["labels_list"]
            except:
                label_list = None
            if 'choice' in config.text_format:
                if '/' in dataset_name:
                    dn_split = dataset_name.split('/')
                    d_name = dn_split[0]
                    config_name = dn_split[1]
                    try:
                        promptList = DatasetTemplates(d_name, config_name)[prompt_name]
                    except:
                        promptList = DatasetTemplates(dataset_name)[prompt_name]
                else:
                    promptList = DatasetTemplates(dataset_name)[prompt_name]
                jinja_prompt = clease_jinja(promptList.jinja)
                answer_choices_format = promptList.answer_choices
            else:
                answer_choices_format = ''
            input_ = get_text_format(text, answer_choices_format, label_list)
            if config.method=='BM25':
                corpus.append(input_)
            else:
                embedding = all_embeddings[train_cnt]
                #train_index[input_] = embedding      
                train_index[train_cnt] = embedding     
                #label_list = row["labels_list"]
            text_to_expert[train_cnt] = f'{dataset_name},{prompt_name}'
            train_cnt+=1

if config.method=='BM25':
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)


print(f'Starting getting 32 stats for eval..!')
eval_cnt = 0
seen_prompt = {}
stats = {}
entries = []
task_level = {}
for key in eval_dict:
    original_dataset_name = key
    dataset_name = format_dataset_name(key)
    samples = eval_dict[key]
    print(dataset_name, len(samples))
    eval_cnt+=len(samples)
    id = 0
    accum_score = 0
    if len(samples) == 0:
        continue
    
    for row in samples:
        prompt_name = row["prompt"]
        if f'{original_dataset_name}@{prompt_name}' not in seen_prompt:
            id = 0
            seen_prompt[f'{original_dataset_name}@{prompt_name}'] = 0       
        if id==32:
            sorted_stats = sorted(stats, key=stats.get, reverse=True)
            print(f'current eval dataset / prompt: {original_dataset_name}@{prompt_name}')
            picked_expert = sorted_stats[0]
            #print(f'picked_expert: {picked_expert}')
            task_level[f'{original_dataset_name}@{prompt_name}'] = sorted_stats
            stats = {}
            id+=1
            continue
        elif id>32:
            continue
        text = row["source"]
        target = row["target"]
        if 'choice' in config.text_format:
            if '/' in dataset_name:
                dn_split = dataset_name.split('/')
                d_name = dn_split[0]
                config_name = dn_split[1]
                try:
                    promptList = DatasetTemplates(d_name, config_name)[prompt_name]
                except:
                    promptList = DatasetTemplates(dataset_name)[prompt_name]
            else:
                promptList = DatasetTemplates(dataset_name)[prompt_name]
            jinja_prompt = clease_jinja(promptList.jinja)
            answer_choices_format = promptList.answer_choices
        else:
            answer_choices_format = ''
        label_list = row["labels_list"]
        query = get_text_format(text, answer_choices_format, label_list)

        if 'ST' in config.method or 'T0' in config.method:
            if config.method=='T0':
                input_ids = tokenizer(text, return_tensors='pt').input_ids
                embedding = t0_retriever(input_ids.to('cuda'))[0][0]
                embedding = torch.mean(embedding, dim=0)  
                eval_embedding = embedding.cpu().data.numpy()
            else:
                eval_embedding = sentence_transformer.encode(query)
            max = 0
            dot_product_lst = []
            dot_to_text = {}
            for train_key in train_index:
                key_embedding = train_index[train_key]             
                dot_product = np.dot(key_embedding, eval_embedding)
                dot_product_lst.append(dot_product)
                if dot_product in dot_to_text:
                    if random.randint(0,100)==0:
                        dot_to_text[dot_product] = f'{text_to_expert[train_key]}'
                else:
                    dot_to_text[dot_product] = f'{text_to_expert[train_key]}'
            dot_product_lst.sort(reverse=True)
            
            expert_cnt  = {}
            for i in range(config.top_k):
                selected_expert = dot_to_text[dot_product_lst[i]]
                if selected_expert not in expert_cnt:
                    expert_cnt[selected_expert]=1
                else:
                    expert_cnt[selected_expert]+=1
            
            # Choosing the expert with the most frequency among the Top K
            max = 0
            for key in expert_cnt:
                if expert_cnt[key]>max:
                    max = expert_cnt[key]
                    expert_name_prompt = key 
            expert_name_prompt = dot_to_text[dot_product_lst[0]]
            expert_name_prompt = expert_name_prompt.split(',')
            expert_dataset = revert_dataset_name(expert_name_prompt[0])
            expert_prompt = expert_name_prompt[1]
            if f'{expert_dataset}@{expert_prompt}'=='ag_news@which_section':
                expert_prompt = 'which_section_choices'
        elif config.method=="BM25":
            tokenized_query = query.split(" ")
            retrieved_train_key = bm25.get_top_n(tokenized_query, corpus, n=1)[0]
            max_prompt = text_to_expert[retrieved_train_key]
            max_prompt_s = max_prompt.split(',')
            expert_dataset = revert_dataset_name(max_prompt_s[0])
            expert_prompt = max_prompt_s[1]
        else:
            raise Exception(f'{config.method} not implemented..')
        key = f'{expert_dataset}@{expert_prompt}@{original_dataset_name}@{prompt_name}@{id}'
        #print(id, key)
        id+=1
        if f'{expert_dataset}@{expert_prompt}' not in stats:
            stats[f'{expert_dataset}@{expert_prompt}']=1
        else:
            stats[f'{expert_dataset}@{expert_prompt}']+=1

with open(f'topk_merge_mappings/{config.method}_{config.text_format}_top{config.top_k}_32.json', "w") as outfile:
    json.dump(task_level, outfile)