import pandas
import os

request_task = input('Enter : ')
request_prompt = input('Enter : ')

result = {}
overall_accuracy=0.0
for filename in os.listdir(f'./output_logs_task_vector/ablation/{request_task}'):
#for filename in os.listdir(f'./output_logs_task_full_lm2'):
    train_task = str(filename).split('-')[0].split('*')[0].replace('/',' ').replace('-',' ')
    try:
        train_prompt = str(filename).split('-')[0].split('*')[1].replace('/',' ').replace('-',' ')
        target_task = str(filename).split('-')[1].split('*')[0].replace('/',' ').replace('-',' ')
        target_prompt = str(filename).split('-')[1].split('*')[1].replace('/',' ').replace('-',' ').replace('.txt','')
        
        if(train_task == request_task) and (train_prompt == request_prompt):
            with open(f'./output_logs_task_vector/ablation/{request_task}/{request_task}*{train_prompt}-{target_task}*{target_prompt}.txt','r') as f:
                for row in f:
                    if row[0:2] != ">>":
                        continue
                    else:
                        accuracy = float(row.split('accuracy : ')[1].split(' | ')[0].strip())
                        overall_accuracy += accuracy
                        result[target_task+"/"+target_prompt] = accuracy
                        break

    except:
        continue
result['overall'] = overall_accuracy/120
print(result['overall'])
x = pandas.DataFrame(data=result,index=['results'])
            
x = x[['overall',
        'hellaswag/complete_first_then',
        'hellaswag/Randomized prompts template',
        #'hellaswag/Appropriate continuation   Yes or No',
        'hellaswag/Predict ending with hint',
        #'hellaswag/Reversed appropriate continuation   Yes or No',
        'hellaswag/how_ends',
        'hellaswag/if_begins_how_continues',
        "story_cloze/Answer Given options",
        "story_cloze/Choose Story Ending",
        "story_cloze/Movie What Happens Next",
        "story_cloze/Story Continuation and Options",
        "story_cloze/Novel Correct Ending",
        "anli_r1/MNLI crowdsource",
        "anli_r1/should assume",
        "anli_r1/does it follow that",
        "anli_r1/GPT 3 style",
        "anli_r1/based on the previous passage",
        "anli_r1/justified in saying",
        "anli_r1/take the following as truth",
        "anli_r1/must be true",
        "anli_r1/can we infer",
        "anli_r1/guaranteed possible impossible",
        "anli_r1/always sometimes never",
        "anli_r1/does this imply",
        "anli_r1/consider always sometimes never",
        "anli_r1/claim true false inconclusive",
        "anli_r1/guaranteed true",
        "anli_r2/MNLI crowdsource",
        "anli_r2/should assume",
        "anli_r2/does it follow that",
        "anli_r2/GPT 3 style",
        "anli_r2/based on the previous passage",
        "anli_r2/justified in saying",
        "anli_r2/take the following as truth",
        "anli_r2/must be true",
        "anli_r2/can we infer",
        "anli_r2/guaranteed possible impossible",
        "anli_r2/always sometimes never",
        "anli_r2/does this imply",
        "anli_r2/consider always sometimes never",
        "anli_r2/claim true false inconclusive",
        "anli_r2/guaranteed true",
        "anli_r3/MNLI crowdsource",
        "anli_r3/should assume",
        "anli_r3/does it follow that",
        "anli_r3/GPT 3 style",
        "anli_r3/based on the previous passage",
        "anli_r3/justified in saying",
        "anli_r3/take the following as truth",
        "anli_r3/must be true",
        "anli_r3/can we infer",
        "anli_r3/guaranteed possible impossible",
        "anli_r3/always sometimes never",
        "anli_r3/does this imply",
        "anli_r3/consider always sometimes never",
        "anli_r3/claim true false inconclusive",
        "anli_r3/guaranteed true",
        "super_glue_copa/exercise",
        #"super_glue_copa/\u2026What could happen next, C1 or C2?",
        "super_glue_copa/i_am_hesitating",
        "super_glue_copa/plausible_alternatives",
        "super_glue_copa/C1 or C2? premise, so because\u2026",
        #"super_glue_copa/\u2026As a result, C1 or C2?",
        "super_glue_copa/best_option",
        #"super_glue_copa/\u2026which may be caused by",
        "super_glue_copa/more likely",
        "super_glue_copa/cause_effect",
        #"super_glue_copa/\u2026why? C1 or C2",
        "super_glue_copa/choose",
        "super_glue_cb/can we infer",
        "super_glue_cb/based on the previous passage",
        "super_glue_cb/claim true false inconclusive",
        "super_glue_cb/does it follow that",
        "super_glue_cb/justified in saying",
        "super_glue_cb/always sometimes never",
        "super_glue_cb/GPT 3 style",
        "super_glue_cb/consider always sometimes never",
        "super_glue_cb/guaranteed true",
        "super_glue_cb/must be true",
        "super_glue_cb/guaranteed possible impossible",
        "super_glue_cb/does this imply",
        "super_glue_cb/MNLI crowdsource",
        "super_glue_cb/should assume",
        "super_glue_cb/take the following as truth",
        "super_glue_rte/MNLI crowdsource",
        "super_glue_rte/guaranteed true",
        "super_glue_rte/can we infer",
        "super_glue_rte/GPT 3 style",
        "super_glue_rte/does this imply",
        "super_glue_rte/should assume",
        "super_glue_rte/does it follow that",
        "super_glue_rte/based on the previous passage",
        "super_glue_rte/justified in saying",
        "super_glue_rte/must be true",
        "super_glue_wsc.fixed/does the pronoun refer to",
        "super_glue_wsc.fixed/by p they mean",
        "super_glue_wsc.fixed/in other words",
        "super_glue_wsc.fixed/I think they mean",
        "super_glue_wsc.fixed/does p stand for",
        "super_glue_wsc.fixed/GPT 3 Style",
        "super_glue_wsc.fixed/replaced with",
        "super_glue_wsc.fixed/p is are r",
        "super_glue_wsc.fixed/the pronoun refers to",
        "super_glue_wsc.fixed/Who or what is are",
        "super_glue_wic/question context meaning with label",
        "super_glue_wic/question context meaning",
        "super_glue_wic/grammar_homework",
        "super_glue_wic/affirmation_true_or_false",
        "super_glue_wic/GPT 3 prompt",
        "super_glue_wic/same_sense",
        "super_glue_wic/question context",
        "super_glue_wic/GPT 3 prompt with label",
        "super_glue_wic/polysemous",
        "super_glue_wic/similar sense",
        "winogrande/does underscore refer to",
        "winogrande/stand for",
        "winogrande/underscore refer to",
        "winogrande/fill in the blank",
        #"winogrande/True or False",
        "winogrande/Replace"
        #'super_glue_copa/cause_effect rouge',
        #'lama/none accuracy',
        #'lambada/please next word rouge'
    ]]

#x.to_csv(f'./csv_results_qqp/{request_task}/{request_prompt}.csv',index=False)
x.to_csv(f'./csv_results_task_vector/ablation/{request_task}-{request_prompt}.csv',index=False)