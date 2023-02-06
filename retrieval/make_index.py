import json
import pprint 
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel, AutoModel, AutoTokenizer
import torch
#from promptsource.promptsource.templates import DatasetTemplates
from promptsource.promptsource.templates import DatasetTemplates
import numpy as np
import argparse
from argparse import ArgumentParser
import re


train_dict = json.load(open("./retrieval_data/seen_target_train_100.json","r"))


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
    model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='ST_12':
    model = SentenceTransformer('all-MiniLM-L12-v2',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='ST_MP':
    model = SentenceTransformer('all-mpnet-base-v2',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='ST_MP_NLI':
    model = SentenceTransformer('nli-mpnet-base-v2',device='cuda:0')
    model.max_seq_length = 256
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
    model.max_seq_length = 256
elif config.method=='ST5_XL':
    model = SentenceTransformer('sentence-transformers/sentence-t5-xl',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='INSTRUCTOR_Large':
    tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-large')
    model = T5EncoderModel.from_pretrained('hkunlp/instructor-large')
    model.to('cuda:0')
    #model.max_seq_length = 256
elif config.method=='INSTRUCTOR_XL':
    tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl')
    model = T5EncoderModel.from_pretrained('hkunlp/instructor-xl')
    model.to('cuda:0')
    #model.max_seq_length = 256
elif config.method=='GTR_Large':
    model = SentenceTransformer('sentence-transformers/gtr-t5-large',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='GTR_XL':
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl',device='cuda:0')
    model.max_seq_length = 256
elif config.method=='DIFFCSE':
    tokenizer = AutoTokenizer.from_pretrained('voidism/diffcse-roberta-base-trans')
    model = AutoModel.from_pretrained('voidism/diffcse-roberta-base-trans')
    model.to('cuda:0')
elif config.method=='T0':
    model = T5EncoderModel.from_pretrained("google/t5-small-lm-adapt")
    t0_small = torch.load("rospr/T0_small.pt")['state_dict']
    t0_new={}
    for key, value in t0_small.items():
        if 'model' in key:
            t0_new[key[6:]] = value
    model.load_state_dict(t0_new, strict=False)
    for par in model.parameters():
        par.requires_grad = False
    model.to('cuda:0')
    model.eval()
    #state_dict = torch.load("rospr/T0_small.pt")
    #model = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
    #model = model._load_state_dict_into_model(model, state_dict, "rospr/T0_small.pt")[0]
    #model.to('cuda:0')
    #model = model.get_encoder()
    tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")
elif config.method=='BM25':
    raise Exception('Not yet supporting BM25 for instance level retrieval')
else:
    raise Exception(f'{config.method} not yet implemented.')

def clease_jinja(jinja):
    jinja = re.sub(r'{{.+?}}', '', jinja)
    jinja = re.sub(r'{.+?}', '', jinja)
    jinja = jinja.replace("|||", "")
    jinja = jinja.replace("\n", "")
    return jinja

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
    else:
        return dataset_name

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

all_embeddings = []
train_cnt = 0
print(f'Getting the training index ready..!')

for key in train_dict:
    dataset_name = key
    prompts = train_dict[dataset_name]
    dataset_name = format_dataset_name(dataset_name)
    for prompt_name in prompts:
        samples = prompts[prompt_name]
        print(f'{dataset_name}, {prompt_name}, {len(samples)}, {train_cnt}')
        for row_index in samples:
            row = samples[row_index]
            train_cnt+=1
            text = row["source"]
            target = row["target"]
            try:
                label_list = row["labels_list"]
            except:
                label_list = None
            try:
                promptList = DatasetTemplates(dataset_name)[prompt_name]
                jinja_prompt = clease_jinja(promptList.jinja)
                answer_choices_format = promptList.answer_choices
            except:
                answer_choices_format = None
            text = get_text_format(text, answer_choices_format, label_list)
            if config.method in ['SIMCSE_SUP','SIMCSE_UNSUP','DIFFCSE','T0','INSTRUCTOR_Large','INSTRUCTOR_XL']:
                tokenize = tokenizer.batch_encode_plus([text], return_tensors = 'pt', max_length=256, padding='max_length', truncation=True, add_special_tokens=True)
                embedding = torch.mean(model(input_ids = tokenize['input_ids'].to('cuda:0'), attention_mask = tokenize['attention_mask'].to('cuda:0')).last_hidden_state,dim=1)
                embedding = embedding[0].cpu().data.numpy()
            else:
                embedding = model.encode(text)
            
            all_embeddings.append(embedding)

print(f'Total train samples count: {train_cnt}')
all_embeddings = np.array(all_embeddings)
np.save(f'./indexes_256/seen_target_train_100_{config.method}_{config.text_format}.npy', all_embeddings)