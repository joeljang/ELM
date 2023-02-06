import pandas
import os
from tqdm import tqdm
task_list = {
    "xsum":{
        "config":"none",
        "prompts":[
            "DOC_write_summary_of_above",
            "article_DOC_summary",
            "DOC_how_would_you_rephrase_few_words",
            "college_roommate_asked_DOC_so_I_recap",
            "DOC_boils_down_to_simple_idea_that",
            "summarize_DOC",
            "summarize_this_DOC_summary",
            "DOC_given_above_write_one_sentence",
            "read_below_DOC_write_abstract",
            "DOC_tldr"
        ]
    },
    "trec":{
        "config":"none",
        "prompts":[
            "what_category_best_describe",
            "fine_grained_LOC",
            "fine_grained_NUM_context_first",
            "fine_grained_ENTY",
            "fine_grained_NUM",
            "pick_the_best_descriptor",
            "fine_grained_open_context_first",
            "fine_grained_LOC_context_first",
            "which_category_best_describes",
            "fine_grained_DESC",
            "trec1",
            "fine_grained_ABBR",
            "fine_grained_ABBR_context_first",
            "trec2",
            "fine_grained_HUM",
            "fine_grained_open",
            "fine_grained_HUM_context_first",
            "fine_grained_DESC_context_first"
        ]
    },
    "cos_e":{
        "config":"v1.11",
        "prompts":[
            "question_description_option_text",
            "question_description_option_id",
            "rationale",
            "question_option_description_text",
            "aligned_with_common_sense",
            "description_question_option_id",
            "explain_why_human",
            "generate_explanation_given_text",
            "description_question_option_text",
            "i_think",
            "question_option_description_id"
        ]
    },
    "commonsense_qa":{
        "config":"none",
        "prompts":[
            "answer_given_question_without_options",
            "question_answering",
            "question_to_answer_index",
            "most_suitable_answer",
            "answer_to_question"
        ]
    },
    "dream":{
        "config":"none",
        "prompts":[
            "generate-last-utterance",
            "answer-to-dialogue",
            "generate-first-utterance",
            "baseline",
            "read_the_following_conversation_and_answer_the_question"
        ]
    },
    "quail":{
        "config":"none",
        "prompts":[
            "context_question_answer_description_id",
            "context_question_answer_description_text",
            "description_context_question_answer_id",
            "context_question_description_answer_text",
            "context_question_description_text",
            "context_description_question_text",
            "context_question_description_answer_id",
            "no_prompt_id",
            "context_description_question_answer_id",
            "description_context_question_text",
            "no_prompt_text",
            "context_description_question_answer_text",
            "description_context_question_answer_text"
        ]
    },
    "quartz":{
        "config":"none",
        "prompts":[
            "use_info_from_question_paragraph",
            "paragraph_question_plain_concat",
            "use_info_from_paragraph_question",
            "answer_question_based_on",
            "answer_question_below",
            "read_passage_below_choose",
            "having_read_above_passage",
            "given_the_fact_answer_the_q"
        ]
    },
    "social_i_qa":{
        "config":"none",
        "prompts":[
            "I was wondering",
            "Show choices and generate answer",
            "Check if a random answer is valid or not",
            "Generate the question from the answer",
            "Generate answer",
            "Show choices and generate index"
        ]
    },
    "wiqa":{
        "config":"none",
        "prompts":[
            "what_might_be_the_first_step_of_the_process",
            "what_might_be_the_last_step_of_the_process",
            "what_is_the_missing_first_step",
            "what_is_the_final_step_of_the_following_process",
            "effect_with_string_answer",
            "which_of_the_following_is_the_supposed_perturbation",
            "effect_with_label_answer",
            "does_the_supposed_perturbation_have_an_effect"
        ]
    },
    "cosmos_qa":{
        "config":"none",
        "prompts":[
            "context_answer_to_question",
            "description_context_question_answer_text",
            "description_context_question_text",
            "description_context_question_answer_id",
            "context_description_question_answer_text",
            "no_prompt_id",
            "context_question_description_text",
            "no_prompt_text",
            "context_description_question_answer_id",
            "context_question_description_answer_id",
            "context_description_question_text",
            "context_question_description_answer_text",
            "only_question_answer"
        ]
    },
    "qasc":{
        "config":"none",
        "prompts":[
            "is_correct_1",
            "qa_with_separated_facts_1",
            "qa_with_separated_facts_3",
            "qa_with_separated_facts_4",
            "qa_with_separated_facts_5",
            "qa_with_combined_facts_1",
            "is_correct_2",
            "qa_with_separated_facts_2"
        ]
    },
    "quarel":{
        "config":"none",
        "prompts":[
            "do_not_use",
            "logic_test",
            "heres_a_story",
            "choose_between",
            "testing_students"
        ]
    },
    "sciq":{
        "config":"none",
        "prompts":[
            "Direct Question (Closed Book)",
            "Multiple Choice (Closed Book)",
            "Multiple Choice Question First",
            "Multiple Choice",
            "Direct Question"
        ]
    },
    "wiki_hop":{
        "config":"original",
        "prompts":[
            "choose_best_object_interrogative_1",
            "explain_relation",
            "generate_object",
            "generate_subject",
            "choose_best_object_affirmative_1",
            "choose_best_object_affirmative_3",
            "generate_subject_and_object",
            "choose_best_object_affirmative_2",
            "choose_best_object_interrogative_2"
        ]
    },
    "amazon_polarity":{
        "config":"none",
        "prompts":[
            "Is_this_review",
            "User_recommend_this_product",
            "Is_this_product_review_positive",
            "Is_this_review_negative",
            "convey_negative_or_positive_sentiment",
            "negative_or_positive_tone",
            "user_satisfied",
            "would_you_buy",
            "flattering_or_not"
        ]
    },
    "app_reviews":{
        "config":"none",
        "prompts":[
            "categorize_rating_using_review",
            "generate_review",
            "convert_to_star_rating",
            "convert_to_rating"
        ]
    },
    "imdb":{
        "config":"none",
        "prompts":[
            "Movie Expressed Sentiment 2",
            "Reviewer Opinion bad good choices",
            "Sentiment with choices ",
            "Reviewer Sentiment Feeling",
            "Writer Expressed Sentiment",
            "Movie Expressed Sentiment",
            "Text Expressed Sentiment",
            "Negation template for positive and negative",
            "Reviewer Enjoyment Yes No",
            "Reviewer Expressed Sentiment",
            "Reviewer Enjoyment"
        ]
    },
    "rotten_tomatoes":{
        "config":"none",
        "prompts":[
            "Movie Expressed Sentiment 2",
            "Reviewer Opinion bad good choices",
            "Sentiment with choices ",
            "Reviewer Sentiment Feeling",
            "Writer Expressed Sentiment",
            "Movie Expressed Sentiment",
            "Text Expressed Sentiment",
            "Reviewer Enjoyment Yes No",
            "Reviewer Expressed Sentiment",
            "Reviewer Enjoyment"
        ]
    },
    "yelp_review_full":{
        "config":"none",
        "prompts":[
            "so_i_would",
            "based_on_that",
            "format_star",
            "this_place",
            "format_score",
            "on_a_scale",
            "format_rating"
        ]
    },
    "paws":{
        "config":"labeled_final",
        "prompts":[
            "task_description-no-label",
            "Meaning",
            "context-question-no-label",
            "Rewrite-no-label",
            "context-question",
            "Concatenation",
            "paraphrase-task",
            "Concatenation-no-label",
            "Meaning-no-label",
            "PAWS-ANLI GPT3",
            "Rewrite",
            "PAWS-ANLI GPT3-no-label"
        ]
    },
    "glue_qqp":{
        "config":"qqp",
        "prompts":[
            "quora",
            "duplicate or not",
            "same thing",
            "answer",
            "meaning",
            "duplicate"
        ]
    },
    "glue_mrpc":{
        "config":"mrpc",
        "prompts":[
            "generate_paraphrase",
            "want to know",
            "paraphrase",
            "equivalent",
            "generate_sentence",
            "replace",
            "same thing"
        ]
    },
    "ag_news":{
        "config":"none",
        "prompts":[
            "classify_question_first",
            "classify_with_choices_question_first",
            "recommend",
            "which_section_choices",
            "which_section",
            "classify_with_choices",
            "classify"
        ]
    },
    "dbpedia_14":{
        "config":"none",
        "prompts":[
            "given_list_what_category_does_the_paragraph_belong_to",
            "pick_one_category_for_the_following_text",
            "given_a_choice_of_categories ",
            "given_a_list_of_category_what_does_the_title_belong_to"
        ]
    },
    "adversarial_qa":{
        "config":"adversarialQA",
        "prompts":[
            "generate_question",
            "tell_what_it_is",
            "question_context_answer",
            "based_on",
            "answer_the_following_q"
        ]
    },
    "quoref":{
        "config":"none",
        "prompts":[
            "Guess Answer",
            "Answer Question Given Context",
            "Find Answer",
            "Context Contains Answer",
            "Given Context Answer Question",
            "What Is The Answer",
            "Answer Test",
            "Guess Title For Context",
            "Found Context Online",
            "Answer Friend Question",
            "Read And Extract "
        ]
    },
    "ropes":{
        "config":"none",
        "prompts":[
            "prompt_beginning",
            "prompt_bottom_no_hint",
            "prompt_bottom_hint_beginning",
            "given_background_situation",
            "plain_no_background",
            "plain_bottom_hint",
            "plain_background_situation",
            "background_new_situation_answer",
            "background_situation_middle",
            "new_situation_background_answer",
            "prompt_mix",
            "read_background_situation"
        ]
    },
    "duorc":{
        "config":"ParaphraseRC",
        "prompts":[
            "build_story_around_qa",
            "decide_worth_it",
            "question_answering",
            "movie_director",
            "generate_question",
            "extract_answer",
            "title_generation",
            "answer_question",
            "generate_question_by_answer"
        ]
    },
    "hotpot_qa":{
        "config":"fullwiki",
        "prompts":[
            "generate_answer_affirmative",
            "classify_question_type",
            "generate_title_affirmative",
            "generate_question",
            "generate_explanations_affirmative",
            "generate_answer_interrogative"
        ]
    },
    "wiki_qa":{
        "config":"none",
        "prompts":[
            "Is This True?",
            "automatic_system",
            "Jeopardy style",
            "Topic Prediction - Question and Answer Pair",
            "Generate Question from Topic",
            "found_on_google",
            "Topic Prediction - Question Only",
            "exercise",
            "Decide_good_answer",
            "Topic Prediction - Answer Only",
            "Direct Answer to Question"
        ]
    },
    "common_gen":{
        "config":"none",
        "prompts":[
            "Given concepts - type 2",
            "Put together",
            "choice in concept centric sentence generation",
            "random task template prompt",
            "topics from the sentence",
            "sentence to concepts",
            "topic to sentence",
            "Example prompt",
            "Given concepts type 1",
        ]
    },
    "wiki_bio":{
        "config":"none",
        "prompts":[
            "who"
        ]
    },
    "cnn_dailymail":{
        "config":"3.0.0",
        "prompts":[
            "write_an_outline",
            "news_summary",
            "2_or_3_sentences",
            "tldr_summary",
            "news_card_view",
            "generate_story",
            "sum_in_brief",
            "news_stock",
            "spice_up_story"
        ]
    },
    "gigaword":{
        "config":"none",
        "prompts":[
            "generate_summary_for_this",
            "reverse_writing",
            "make_a_title",
            "first_sentence_title",
            "TLDR",
            "write_its_sentence",
            "write_a_title_for_this_sentence",
            "in_a_nutshell",
            "write_an_article"
        ]
    },
    "multi_news":{
        "config":"none",
        "prompts":[
            "what are the key points",
            "synthesize",
            "summary scenario",
            "summarize",
            "expand (reverse task)",
            "distill"
        ]
    },
    "samsum":{
        "config":"none",
        "prompts":[
            "Summarize this dialogue:",
            "Given the above dialogue write a summary",
            "Summarize:",
            "To sum up this dialog",
            "Generate a summary for this dialogue",
            "Write a dialogue that match this summary",
            "Sum up the following dialogue"
        ]
    }
}

target_task_list = {
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
bleu_task = ['asset','wiki_auto']
bert_score_task = ['covid_qa','eli5','emdg','esnli','twitter']
rouge_task = ['ct0_gigaword','haiku']

for request_task in tqdm(task_list):
    for request_prompt in task_list[request_task]['prompts']:
        request_task = request_task.replace('/',' ').replace('-',' ')
        request_prompt = request_prompt.replace('/',' ').replace('-',' ')
        result = {}
        overall_accuracy=0.0
        for filename in os.listdir(f'./output_logs_target_eval/{request_task}'):
        #for filename in os.listdir(f'./output_logs_task_full_lm2'):
            train_task = str(filename).split('-')[0].split('*')[0].replace('/',' ').replace('-',' ')
            try:
                train_prompt = str(filename).split('-')[0].split('*')[1].replace('/',' ').replace('-',' ')
                target_task = str(filename).split('-')[1].split('*')[0].replace('/',' ').replace('-',' ')
                target_prompt = str(filename).split('-')[1].split('*')[1].replace('/',' ').replace('-',' ').replace('.txt','')
                
                if(train_task == request_task) and (train_prompt == request_prompt):
                    with open(f'./output_logs_target_eval/{request_task}/{request_task}*{train_prompt}-{target_task}*{target_prompt}.txt','r') as f:
                        for row in f:
                            if row[0:2] != ">>":
                                continue
                            else:
                                if target_task in bleu_task:
                                    bleu = float(row.split('bleu : ')[1].split(' | ')[0].strip())
                                    overall_accuracy += bleu
                                    result[target_task+"/"+target_prompt] = bleu
                                    break
                                elif target_task in rouge_task:
                                    rouge = float(row.split('rouge1 : ')[1].split(' | ')[0].strip())
                                    overall_accuracy += rouge
                                    result[target_task+"/"+target_prompt] = rouge
                                    break
                                elif target_task in bert_score_task:
                                    bert_score = float(row.split('bertscore : ')[1].split(' | ')[0].strip())*100
                                    overall_accuracy += bert_score
                                    result[target_task+"/"+target_prompt] = bert_score
                                    break
            except:
                continue
        result['overall'] = overall_accuracy/16
        print(result['overall'])
        x = pandas.DataFrame(data=result,index=['results'])
                    
        x = x[['overall',
                'wiki_auto/simplification_1',
                'wiki_auto/simplification_2',
                'asset/simplification_1',
                'asset/simplification_2',
                'ct0_gigaword/constrain_contain+make_a_title',
                'ct0_gigaword/constrain_contain+write_its_sentence',
                'ct0_gigaword/constrain_start+make_a_title',
                'ct0_gigaword/constrain_start+write_its_sentence',
                'ct0_gigaword/constrain_end+make_a_title',
                'ct0_gigaword/constrain_end+write_its_sentence',
                'haiku/do_nothing',
                'covid_qa/covid_qa_deepset',
                'eli5/generate_a_question_1',
                'emdg/dialogue_with_emotion',
                'esnli/explain_why',
                'twitter/tweet_as+about'
            ]]

        #x.to_csv(f'./csv_results_qqp/{request_task}/{request_prompt}.csv',index=False)
        x.to_csv(f'./csv_results_target_eval/{request_task}-{request_prompt}.csv',index=False)