from data.tasks import AutoTask, TASK_MAPPING, STORYCLOZE, ANLIR1, ANLIR2, ANLIR3, WIKIHOP, AMAZONPOLARITY, YELPREVIEWFULL, DBPEDIA14, TREC, IMDB, APPREVIEWS, GIGAWORD, ROTTENTOMATOES, GeneralTask
import torch
import json
task_list = {
    "wiki_auto":{
        "config":"skip",
        "prompts":[
            "simplification_1",
            "simplification_2",
        ]
    },
    "asset":{
        "config":"skip",
        "prompts":[
            "simplification_1",
            "simplification_2",
        ]
    },
    "ct0_gigaword":{
        "config":"skip",
        "prompts":[
        "constrain_contain+make_a_title",
        "constrain_contain+write_its_sentence",
        "constrain_end+make_a_title",
        "constrain_end+write_its_sentence",
        "constrain_start+make_a_title",
        "constrain_start+write_its_sentence",
        ]
    },
    "haiku":{
        "config":"skip",
        "prompts":[
            "do_nothing"
        ]
    },
    "covid_qa":{
        "config":"skip",
        "prompts":[
            "covid_qa_deepset"
        ]
    },
    "eli5":{
        "config":"skip",
        "prompts":[
            "generate_a_question_1"
        ]
    },
    "emdg":{
        "config":"skip",
        "prompts":[
            "dialogue_with_emotion"
        ]
    },
    "esnli":{
        "config":"skip",
        "prompts":[
            "explain_why"
        ]
    },
    "twitter":{
        "config":"skip",
        "prompts":[
            "tweet_as+about"
        ]
    }
}

def shuffled_indices(dataset):
    num_samples = len(dataset)
    generator = torch.Generator()
    generator.manual_seed(42)
    return torch.randperm(num_samples, generator=generator).tolist()

column_names = ['source', 'target', 'extra_fields']
max_num_instances = 300
n_obs = max_num_instances
eval_instances={}
for _,task_name in enumerate(task_list):
    n_obs=max_num_instances
    task = task_list[task_name]
    config = task['config']
    
    eval_instances[task_name] = {}
    for prompt in task['prompts']:
        data_class = AutoTask.get(task_name, config, prompt)
        
        if prompt not in eval_instances[task_name].keys():
            eval_instances[task_name][prompt] = {}
        eval_data = data_class.load_dataset('validation')
        
        if len(eval_data)< max_num_instances:
            n_obs=len(eval_data)
        eval_indices = shuffled_indices(eval_data)
        eval_indices = eval_indices[:n_obs]
        eval_data = eval_data.select(eval_indices)


        for idx,data in enumerate(eval_data):
            d = data_class.preprocessor(data)
            if 'labels_list' in d:
                e_formatted_d = {
                        "config" : config,
                        "task" : d['task'],
                        "prompt" : prompt,
                        "source" : d['source'],
                        "target" : d['target'],
                        "labels_list" : d['labels_list']
                    }
            else:
                e_formatted_d = {
                    "config" : config,
                    "prompt" : prompt,
                    "source" : d['source'],
                    "target" : d['target']
                }
            eval_instances[task_name][prompt][str(idx)]=e_formatted_d
            
print(len(eval_instances))
print(len(eval_instances['emdg']))
        
with open(f'target_eval_{max_num_instances}.json','w') as f:
    json.dump(eval_instances,f,indent=4)