from data.tasks import AutoTask, TASK_MAPPING, STORYCLOZE, ANLIR1, ANLIR2, ANLIR3, WIKIHOP, AMAZONPOLARITY, YELPREVIEWFULL, DBPEDIA14, TREC, IMDB, APPREVIEWS, GIGAWORD, ROTTENTOMATOES, GeneralTask
import torch
import json
task_list = {
    "hellaswag":{
        "config":"none",
        "prompts":[
            "complete_first_then",
            "Randomized prompts template",
            "Predict ending with hint",
            "how_ends",
            "if_begins_how_continues",
        ]
    },
    "story_cloze":{
        "config":"none",
        "prompts":[
            "Answer Given options",
            "Choose Story Ending",
            "Movie What Happens Next",
            "Story Continuation and Options",
            "Novel Correct Ending",
        ]
    },
    "anli_r1":{
        "config":"dev_r1",
        "prompts":[
        "MNLI crowdsource",
        "should assume",
        "does it follow that",
        "GPT-3 style",
        "based on the previous passage",
        "justified in saying",
        "take the following as truth",
        "must be true",
        "can we infer",
        "guaranteed/possible/impossible",
        "always/sometimes/never",
        "does this imply",
        "consider always/sometimes/never",
        "claim true/false/inconclusive",
        "guaranteed true",
        ]
    },
    "anli_r2":{
        "config":"dev_r2",
        "prompts":[
            "MNLI crowdsource",
            "should assume",
            "does it follow that",
            "GPT-3 style",
            "based on the previous passage",
            "justified in saying",
            "take the following as truth",
            "must be true",
            "can we infer",
            "guaranteed/possible/impossible",
            "always/sometimes/never",
            "does this imply",
            "consider always/sometimes/never",
            "claim true/false/inconclusive",
            "guaranteed true",
        ]
    },
    "anli_r3":{
        "config":"dev_r3",
        "prompts":[
            "MNLI crowdsource",
            "should assume",
            "does it follow that",
            "GPT-3 style",
            "based on the previous passage",
            "justified in saying",
            "take the following as truth",
            "must be true",
            "can we infer",
            "guaranteed/possible/impossible",
            "always/sometimes/never",
            "does this imply",
            "consider always/sometimes/never",
            "claim true/false/inconclusive",
            "guaranteed true",
        ]
    },
    "super_glue_copa":{
        "config":"copa",
        "prompts":[
            "exercise",
            "i_am_hesitating",
            "plausible_alternatives",
            "C1 or C2? premise, so/because…",
            "best_option",
            "more likely",
            "cause_effect",
            "choose",
        ]
    },
    "super_glue_cb":{
        "config":"cb",
        "prompts":[
            "can we infer",
            "based on the previous passage",
            "claim true/false/inconclusive",
            "does it follow that",
            "justified in saying",
            "always/sometimes/never",
            "GPT-3 style",
            "consider always/sometimes/never",
            "guaranteed true",
            "must be true",
            "guaranteed/possible/impossible",
            "does this imply",
            "MNLI crowdsource",
            "should assume",
            "take the following as truth",
        ]
    },
    "super_glue_rte":{
        "config":"rte",
        "prompts":[
            "MNLI crowdsource",
            "guaranteed true",
            "can we infer",
            "GPT-3 style",
            "does this imply",
            "should assume",
            "does it follow that",
            "based on the previous passage",
            "justified in saying",
            "must be true",
        ]
    },
    "super_glue_wsc.fixed":{
        "config":"wsc.fixed",
        "prompts":[
            "does the pronoun refer to",
            "by p they mean",
            "in other words",
            "I think they mean",
            "does p stand for",
            "GPT-3 Style",
            "replaced with",
            "p is/are r",
            "the pronoun refers to",
            "Who or what is/are",
        ]
    },
    "super_glue_wic":{
        "config":"wic",
        "prompts":[
            "question-context-meaning-with-label",
            "question-context-meaning",
            "grammar_homework",
            "affirmation_true_or_false",
            "GPT-3-prompt",
            "same_sense",
            "question-context",
            "GPT-3-prompt-with-label",
            "polysemous",
            "similar-sense",
        ]
    },
    "winogrande":{
        "config":"winogrande_xl",
        "prompts":[
            "does underscore refer to",
            "stand for",
            "underscore refer to",
            "fill in the blank",
            "Replace"
        ]
    }
}

def shuffled_indices(dataset):
    num_samples = len(dataset)
    generator = torch.Generator()
    generator.manual_seed(42)
    return torch.randperm(num_samples, generator=generator).tolist()

column_names = ['source', 'target', 'extra_fields']
choose_subset = False
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
        if choose_subset:
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
print(len(eval_instances['hellaswag']))
        
with open(f'unseen_eval_{max_num_instances}_FULL.json','w') as f:
    json.dump(eval_instances,f,indent=4)