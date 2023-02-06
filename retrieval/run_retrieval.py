import json
import pprint 
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
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
import random

def euclidean_distance(x,y):
  """ return euclidean distance between two lists """
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

# Loading configs
parser = ArgumentParser()
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--eval_objective', default=None, type=str)
arg_ = parser.parse_args()
if arg_.config is None:
    raise NameError("Include a config file in the argument please.")

# Getting configurations
eval_objective = arg_.eval_objective
config_path = arg_.config
with open(config_path) as config_file:
    config = json.load(config_file)
config = argparse.Namespace(**config)

if config.method=='ST':
    model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='ST_12':
    model = SentenceTransformer('all-MiniLM-L12-v2',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='ST_MP':
    model = SentenceTransformer('all-mpnet-base-v2',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='ST_MP_NLI':
    model = SentenceTransformer('nli-mpnet-base-v2',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='SIMCSE_SUP':
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model.to('cuda:0')
elif config.method=='SIMCSE_UNSUP':
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-large')
    model.to('cuda:0')
elif config.method=='ST5_Large':
    model = SentenceTransformer('sentence-transformers/sentence-t5-large',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='ST5_XL':
    model = SentenceTransformer('sentence-transformers/sentence-t5-xl',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='INSTRUCTOR_Large':
    tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-large')
    model = T5EncoderModel.from_pretrained('hkunlp/instructor-large')
    model.to('cuda:0')
    #model.max_seq_length = 512
elif config.method=='INSTRUCTOR_XL':
    tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl')
    model = T5EncoderModel.from_pretrained('hkunlp/instructor-xl')
    model.to('cuda:0')
    #model.max_seq_length = 512
elif config.method=='GTR_Large':
    model = SentenceTransformer('sentence-transformers/gtr-t5-large',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='GTR_XL':
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl',device='cuda:0')
    model.max_seq_length = 512
elif config.method=='DIFFCSE':
    tokenizer = AutoTokenizer.from_pretrained('voidism/diffcse-roberta-base-trans')
    model = AutoModel.from_pretrained('voidism/diffcse-roberta-base-trans')
    model.to('cuda:0')
elif config.method=='T0':
    state_dict = torch.load("rospr/T0_small.pt")
    model = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
    model = model._load_state_dict_into_model(model, state_dict, "rospr/T0_small.pt")[0]
    model.to('cuda')
    model = model.get_encoder()
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
        text = f"Instance: {instance}"
    elif config.text_format=='label':
        text = f"Answer Choices: {label_list}"
    elif config.text_format=='choice':
        text = f"Answer Choices: {answer_choice_format}"
    elif config.text_format=='label_instance':
        text = f"Answer Choices: {label_list}, Instance: {instance}"
    elif config.text_format=='choice_instance':
        text = f"Answer Choices: {answer_choice_format}, Instance: {instance}"
    elif config.text_format=='instance_new':
        text = f"{instance}"
    elif config.text_format=='label_new':
        text = f"{label_list}"
    elif config.text_format=='choice_new':
        text = f"{answer_choice_format}"
    elif config.text_format=='label_instance_new':
        text = f"{label_list}</s>{instance}"
    elif config.text_format=='choice_instance_new':
        text = f"{answer_choice_format}</s>{instance}"
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
    elif dataset_name=='anli_r1':
        return 'anli_dev_r1'
    elif dataset_name=='anli_r2':
        return 'anli_dev_r2'
    elif dataset_name=='anli_r3':
        return 'anli_dev_r3'
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

train_dict = json.load(open("./retrieval_data/seen_target_train_100.json","r"))
if eval_objective=='unseen':
    eval_dict = json.load(open("./retrieval_data/unseen_eval_300.json","r"))
    df = pd.read_csv('./retrieval_data/unseen_prompt_experts.csv')
elif eval_objective=='seen':
    eval_dict = json.load(open("./retrieval_data/seen_eval_300.json","r"))
    #df = pd.read_csv('./retrieval_data/seen_prompt_experts.csv')
elif eval_objective=='target':
    eval_dict = json.load(open("./retrieval_data/target_eval_300.json","r"))
    df = pd.read_csv('./retrieval_data/target_prompt_experts.csv')

if eval_objective!='seen':
    expert_name_to_scores = {}
    eval_datasets = df.columns[2:]
    eval_dataset_prompts = df.iloc[0][2:]
    print('##############################################')
    print(eval_datasets)
    print()
    print(eval_dataset_prompts)
    print('##############################################')

    for index,row in df.iterrows():
        if index==0:
            #eval_dataset_prompts = row[1:]
            continue
        else:
            #train_dataset = row.name
            print('@@@@@@@@@@@@@@@@@@@@@@')
            print(row)
            print('@@@@@@@@@@@@@@@@@@@@@@')
            train_dataset = row[0]
            train_prompt = row[1]
            print('#################')
            print(str(train_dataset)+", "+str(train_prompt))
            print('#################')
            if train_dataset not in expert_name_to_scores:
                expert_name_to_scores[train_dataset] = {}
            expert_name_to_scores[train_dataset][train_prompt] = row[2:]
        

#if 'ST' in config.method or 'T0' in config.method:
all_embeddings = np.load(f"./indexes_256/seen_target_train_100_{config.method}_{config.text_format}.npy")

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
                try:
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
                except:
                    answer_choices_format = None
            else:
                answer_choices_format = None
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


print(f'Starting trial eval..!')
eval_cnt = 0
seen_prompt = {}
stats = {}
entries = []
task_level = {}
for key in eval_dict:
    original_dataset_name = key
    formated_dataset = format_dataset_name(key)
    if 'anli' in formated_dataset:
        dataset_name = 'anli'
    else:
        dataset_name = formated_dataset
    samples = eval_dict[key]
    print(dataset_name, len(samples))
    eval_cnt+=len(samples)
    id_ = 0
    accum_score = 0
    if len(samples) == 0:
        continue
    for prompt_name in samples:
        # print(samples)
        # print()
        # print(prompt_name)
        # print()
        # print(samples[prompt_name])
        for idx,row in samples[prompt_name].items():
                
            #prompt_name = row["prompt"]
            if f'{original_dataset_name}@{prompt_name}' not in seen_prompt:
                id_ = 0
                seen_prompt[f'{original_dataset_name}@{prompt_name}'] = 0       
            if id_==32:
                stats = sorted(stats, key=stats.get, reverse=True)
                print(f'current eval dataset / prompt: {original_dataset_name}@{prompt_name}')
                picked_expert = stats[0]
                print(f'picked_expert: {picked_expert}')
                print(f'adding to task_level: {formated_dataset}, {prompt_name}')
                if eval_objective=='seen':
                    with open(f'./seen_target_task_check_{config.method}_{config.text_format}_32.txt','a') as f:
                        f.write(f"{original_dataset_name}@{prompt_name} / {picked_expert}\n")
                task_level[f'{formated_dataset}@{prompt_name}'] = picked_expert
                stats = {}
                id_+=1
                continue
            elif id_>32:
                continue
            text = row["source"]
            target = row["target"]
            try:
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
                    answer_choices_format = None
            except:
                answer_choices_format = None
            try:
                label_list = row["labels_list"]
            except:
                label_list = None
            query = get_text_format(text, answer_choices_format, label_list)

            if config.method !='BM25':
                if config.method in ['SIMCSE_SUP','SIMCSE_UNSUP','DIFFCSE','T0','INSTRUCTOR_Large','INSTRUCTOR_XL']:
                    input_ids = tokenizer(text, return_tensors='pt').input_ids
                    embedding = model(input_ids.to('cuda'))[0][0]
                    embedding = torch.mean(embedding, dim=0)  
                    eval_embedding = embedding.cpu().data.numpy()
                else:
                    eval_embedding = model.encode(query)
                max_ = 0
                dot_product_lst = []
                dot_to_text = {}
                for train_key in train_index:
                    key_embedding = train_index[train_key]             
                    dot_product = np.dot(key_embedding, eval_embedding)
                    dot_product_lst.append(dot_product)
                    #if dot_product in dot_to_text:
                    #    if random.randint(0,100)==0:
                    #        dot_to_text[dot_product] = f'{text_to_expert[train_key]}'
                    #else:
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
                max_ = 0
                for key in expert_cnt:
                    if expert_cnt[key]>max_:
                        max_ = expert_cnt[key]
                        expert_name_prompt = key 
                expert_name_prompt = dot_to_text[dot_product_lst[0]]
                expert_name_prompt = expert_name_prompt.split(',')
                expert_dataset = expert_name_prompt[0]
                #expert_dataset = revert_dataset_name(expert_name_prompt[0])
                expert_prompt = expert_name_prompt[1]
                if f'{expert_dataset}@{expert_prompt}'=='ag_news@which_section':
                    expert_prompt = 'which_section_choices'
            elif config.method=="BM25":
                tokenized_query = query.split(" ")
                retrieved_train_key = bm25.get_top_n(tokenized_query, corpus, n=1)[0]
                max_prompt = text_to_expert[retrieved_train_key]
                max_prompt_s = max_prompt.split(',')
                expert_dataset = max_prompt_s[0]
                #expert_dataset = revert_dataset_name(max_prompt_s[0])
                expert_prompt = max_prompt_s[1]
            else:
                raise Exception(f'{config.method} not implemented..')
            id_+=1
            if f'{expert_dataset}@{expert_prompt}' not in stats:
                stats[f'{expert_dataset}@{expert_prompt}']=1
            else:
                stats[f'{expert_dataset}@{expert_prompt}']+=1

print(f'Starting task-level eval..!')
if eval_objective=='unseen':
    evals = ['hellaswag', 'story_cloze/2016', 'anli_dev_r1', 'anli_dev_r2', 'anli_dev_r3', 'super_glue/copa', 'super_glue/cb', 'super_glue/rte', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']
elif eval_objective=='target':
    evals = ['wiki_auto','asset','ct0_gigaword','haiku','covid_qa','eli5','emdg','esnli','twitter']
entries = []
# Evaluate each evaluation prompt
if eval_objective!='seen':
    for i in range(len(eval_datasets)):
        dataset = eval_datasets[i]
        prompt = eval_dataset_prompts[i]
        # if prompt in 'C1 or C2? premise, so/because\u2026':
        #     prompt = '"C1 or C2? premise, so/because\u2026"'
        # Getting the exact dataset name:
        for name in evals:
            if name in dataset:
                dataset = name
        # print('#######################')
        # for key in task_level:
        #     print(key+"     /     "+task_level[key])
        # for key in expert_name_to_scores:
        #     print(key)
        #     print(expert_name_to_scores[key])
        #     print()
        picked_expert_dataset_name = task_level[f'{dataset}@{prompt}'].split('@')[0]
        picked_expert_prompt_name = task_level[f'{dataset}@{prompt}'].split('@')[1]
        
        score = expert_name_to_scores[picked_expert_dataset_name][picked_expert_prompt_name][i]
        entries.append(score)

    print(entries)
    pd.DataFrame([entries]).to_csv(f'results_seen_target_256/{config.method}_{config.text_format}_32.csv', index=False) 
    print(f'good job')