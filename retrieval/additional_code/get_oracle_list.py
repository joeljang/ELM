import pandas as pd 
import json

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

oracle_sorted = {}

evals = ['hellaswag', 'story_cloze/2016', 'anli_dev_r1', 'anli_dev_r2', 'anli_dev_r3', 'super_glue/copa', 'super_glue/cb', 'super_glue/rte', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']
# Evaluate each evaluation prompt
for i in range(len(eval_datasets)):
    eval_dataset = eval_datasets[i]
    eval_prompt = eval_dataset_prompts[i]
    # Getting the exact dataset name:
    for name in evals:
        if name in eval_dataset:
            eval_dataset = name
    max_val = 0
    expert_results = {}
    for j in range(len(train_datasets)):
        train_dataset = train_datasets[j]
        train_prompt = train_prompts[j]
        key = f'{train_dataset}@{train_prompt}'
        val = train_dataset_rows[j][i]
        expert_results[key] = val
    #print(expert_results)
    sorted_expert_results = sorted(expert_results, key=expert_results.get, reverse=True)
    print(f'{eval_dataset}@{eval_prompt}')
    oracle_sorted[f'{eval_dataset}@{eval_prompt}'] = sorted_expert_results

with open(f'oracle_sorted.json', "w") as outfile:
    json.dump(oracle_sorted, outfile)