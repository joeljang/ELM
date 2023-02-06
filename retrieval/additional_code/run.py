import pandas as pd 
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from promptsource.promptsource.templates import DatasetTemplates
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from rank_bm25 import BM25Okapi
from statistics import mean
import re
import datasets

# Setting the method to use
METHOD = 'ST'
#METHOD = 'BM25'
#TEXT_FORMAT = 'prompt'
TEXT_FORMAT = 'label_prompt'
#TEXT_FORMAT = 'label'
#TEXT_FORMAT = 'label@prompt'

def clease_jinja(jinja):
    jinja = re.sub(r'{{.+?}}', '', jinja)
    jinja = re.sub(r'{.+?}', '', jinja)
    jinja = jinja.replace("|||", "")
    jinja = jinja.replace("\n", "")
    return jinja

# Setting the text format to use for retrieval (task level)
def get_text_format(prompt, hard_prompt, answer_choices):
    if TEXT_FORMAT=='prompt':
        text = f"Prompt: {hard_prompt}"
    elif TEXT_FORMAT=='label_prompt':
        text = f"Answer Choices: {answer_choices}, Prompt: {hard_prompt}"
    elif TEXT_FORMAT=='label':
        text = f"Answer Choices: {answer_choices}"
    else:
        raise Exception('Select the correct TEXT_FORMAT')
    return text

from math import sqrt, pow, exp
def squared_sum(x):
  """ return 3 rounded square rooted value """
  return round(sqrt(sum([a*a for a in x])),3)

def euclidean_distance(x,y):
  """ return euclidean distance between two lists """
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def cos_similarity(x,y):
  """ return cosine similarity between two lists """
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

df = pd.read_csv('experts.csv')
eval_datasets = df.columns[2:]

train_datasets = []
train_prompts = []
train_dataset_rows = []
for index,row in df.iterrows():
    if index==0:
        eval_dataset_prompts = row[2:]
    else:
        train_dataset = row['Dataset']
        train_dataset_prompt = row['Prompt']
        train_dataset_rows.append(row[2:])
        train_datasets.append(train_dataset)
        train_prompts.append(train_dataset_prompt)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

print(f'Length of train prompts: {len(train_prompts)}')
print(f'Length of eval prompts: {len(eval_datasets)}')

# Make the Training Index
index = {}
index_values = {}
text_to_prompt = {}
corpus = []
for i in range(len(train_prompts)):
    dataset = train_datasets[i]
    prompt = train_prompts[i]
    promptList = DatasetTemplates(dataset)[prompt]
    hard_prompt = clease_jinja(promptList.jinja)
    answer_choices = promptList.answer_choices
    text = get_text_format(prompt, hard_prompt, answer_choices) # setting format for retrieval
    corpus.append(text)
    embedding = sentence_transformer.encode(text)
    index[f'{dataset},{prompt}'] = embedding
    index_values[f'{dataset},{prompt}'] = train_dataset_rows[i]
    text_to_prompt[text] = f'{dataset},{prompt}'

#index_values['testing,prompt'] = 0
#index['testing,prompt'] = sentence_transformer.encode('Replace the _ in the above sentence with the correct option: - -')

evals = ['hellaswag', 'story_cloze/2016', 'anli_dev_r1', 'anli_dev_r2', 'anli_dev_r3', 'super_glue/copa', 'super_glue/cb', 'super_glue/rte', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']
results = []
index_keys = list(index.keys())
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Evaluate each evaluation prompt
for i in range(len(eval_datasets)):
    dataset = eval_datasets[i]
    prompt = eval_dataset_prompts[i]
    # Getting the exact dataset name:
    for name in evals:
        if name in dataset:
            dataset = name
        if 'anli_dev' in dataset:
            dataset = 'anli'
    promptList = DatasetTemplates(dataset)[prompt]
    hard_prompt = clease_jinja(promptList.jinja)
    answer_choices = promptList.answer_choices
    text = get_text_format(prompt, hard_prompt, answer_choices) # setting format for retrieval

    if METHOD=='ST':
        embedding = sentence_transformer.encode(text)
        max = 0
        products = []
        for key in index:
            key_embedding = index[key]
            #dot_product = np.dot(key_embedding, embedding)
            #dot_product = euclidean_distance(key_embedding, embedding)
            dot_product = cos_similarity(key_embedding, embedding)
            products.append(dot_product)
            if dot_product > max:
                max = dot_product
                max_prompt = key
        
    elif METHOD=='BM25':
        tokenized_query = text.split(" ")
        retrieved_hard_prompt = bm25.get_top_n(tokenized_query, corpus, n=1)[0]
        max_prompt = text_to_prompt[retrieved_hard_prompt]
    elif METHOD=='random':
        max_prompt = random.choice(index_keys)
    else:
        raise Exception('Choose the correct METHOD')
    r_dataset = max_prompt.split(',')[0]
    r_prompt = max_prompt.split(',')[1:]
    r_prompt = "".join(r_prompt)
    r_promptList = DatasetTemplates(r_dataset)[r_prompt]
    r_hard_prompt = clease_jinja(r_promptList.jinja)
    r_answer_choices = r_promptList.answer_choices
    print(f'eval dataset: {dataset}, eval prompt: {prompt}, answer choices: {answer_choices}')
    print(f'retrieved: {max_prompt}, hard_prompt: {r_answer_choices}\n')
    val = index_values[max_prompt][i]
    results.append(val)

pd.DataFrame([results]).to_csv(f'{METHOD}_{TEXT_FORMAT}.csv')
exit()

print(results)
expert_nums = [5, 5, 15, 15, 15, 8, 15, 10, 10, 10, 5]
eval_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cnt = 0
indx = 0
val_sum = 0
for i in range(len(results)):
    val = results[i]
    val_sum += float(val)
    cnt+=1
    if cnt == expert_nums[indx]:
        avg = val_sum / cnt
        eval_results[indx] = avg
        cnt=0
        val_sum=0
        indx+=1
print(f'{METHOD}_{TEXT_FORMAT}')
print(eval_results)
print(f'AVG: {mean(eval_results)}')
