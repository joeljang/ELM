# Code fo making Big Bench and MMLU Evaluation data in .json format
import json
import os 

def dataset_prompt_setting(type_path, dataset_name, dataset_config_name, bigbench_path):
    if type_path == 'bigbench':
        if dataset_config_name==None:
            path = os.path.join(bigbench_path, dataset_name, "task.json")
        else:
            path = os.path.join(bigbench_path, dataset_name, dataset_config_name, "task.json")
        with open(path, 'r') as file:
            data = json.load(file)
        instruction = data["description"]
        example = data["examples"]
        task_prefix = data.get("task_prefix", "")
        input_prefix = data.get("example_input_prefix", "\nQ: ")
        output_prefix = data.get("example_output_prefix", "\nA: ")
        choice_prefix = data.get("choice_prefix", "\n choice: ")
        append_choices_to_input = data.get("append_choices_to_input", True)

        return example, instruction, [task_prefix, input_prefix, output_prefix, choice_prefix, append_choices_to_input]

def bigbench_input(input, options, task_prefix, input_prefix, choice_prefix, append_choices_to_input, output_prefix):
    def choices_string(choices, options, append_choices_to_input):
        if append_choices_to_input:
            choices_string = choices+choices.join(options)
        #choices_string = f'{choices}'+f'{choices}'.join(options)
        else:
            choices_string=""
        return choices_string
    input_ = f'{task_prefix}{input_prefix}{input}{choices_string(choice_prefix, options, append_choices_to_input)}{output_prefix}'           
    return input_

bigbench_tasks_all = ['code_line_description', 
    'conceptual_combinations/contradictions', 'conceptual_combinations/emergent_properties', 'conceptual_combinations/fanciful_fictional_combinations', 'conceptual_combinations/homonyms', 'conceptual_combinations/invented_words', 'conceptual_combinations/homonyms', 'conceptual_combinations/surprising_uncommon_combinations',
    'formal_fallacies_syllogisms_negation', 
    'hindu_knowledge', 
    'known_unknowns', 
    'language_identification', 
    'logic_grid_puzzle', 
    'logical_deduction/five_objects', 'logical_deduction/seven_objects', 'logical_deduction/three_objects', 
    'misconceptions', 
    'movie_dialog_same_or_different', 
    'novel_concepts', 
    'strategyqa', 
    'vitaminc_fact_verification', 
    'winowhy']

bigbench_tasks = ['code_line_description', 
    'conceptual_combinations/contradictions', 'conceptual_combinations/emergent_properties', 'conceptual_combinations/fanciful_fictional_combinations', 'conceptual_combinations/homonyms', 'conceptual_combinations/invented_words', 'conceptual_combinations/homonyms', 'conceptual_combinations/surprising_uncommon_combinations',
    'formal_fallacies_syllogisms_negation', 
    'hindu_knowledge', 
    'known_unknowns', 
    'language_identification', 
    'logic_grid_puzzle', 
    'logical_deduction/five_objects', 'logical_deduction/seven_objects', 'logical_deduction/three_objects', 
    'misconceptions', 
    'movie_dialog_same_or_different', 
    'strategyqa', 
    'vitaminc_fact_verification', 
    'winowhy']


bigbench_path = 'retrieval_data/benchmark_tasks'
evals = {}
for origin_dataset_name in bigbench_tasks:
    if '/' in origin_dataset_name:
        dataset_name_s = origin_dataset_name.split('/')
        dataset_name = dataset_name_s[0]
        dataset_config_name = dataset_name_s[1]
    else:
        dataset_name = origin_dataset_name
        dataset_config_name = None
    eval_dataset = dataset_prompt_setting('bigbench', dataset_name, dataset_config_name, bigbench_path)
    print(f'{origin_dataset_name}, number of examples: {len(eval_dataset[0])}')
    task_prefix, input_prefix, output_prefix, choice_prefix, append_choices_to_input = eval_dataset[2]
    for example in eval_dataset[0]:
        eval_instance = {}
        if dataset_config_name==None:
            eval_instance['config'] = "none"
        else:
            eval_instance['config'] = dataset_config_name
        eval_instance['task'] = dataset_name
        input = example['input']
        eval_instance['label_list'] = []
        eval_instance['target'] = None
        for label in example['target_scores']:
            eval_instance['label_list'].append(label)
            if example['target_scores'][label] == 1:
                if eval_instance['target']!=None:
                    print(example['target_scores'])
                    raise Exception('there are two labels that are correct..!!') 
                eval_instance['target'] = label
        eval_instance['source'] = bigbench_input(input, eval_instance['label_list'], task_prefix, input_prefix, choice_prefix, append_choices_to_input, output_prefix)
        if eval_instance['target']==None:
            raise Exception('there are zero labels with correct answer!')
        if dataset_name not in evals:
            evals[dataset_name] = [eval_instance]
        else:
            evals[dataset_name].append(eval_instance)

print(len(evals))
with open("retrieval_data/bigbench_eval.json", "w") as fp:
    json.dump(evals,fp) 