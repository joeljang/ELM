from collections import OrderedDict
import collections
import abc
import functools
from typing import Callable, List, Mapping
from utils import pad_punctuation
from seq2seq.metrics import metrics
from .utils import round_stsb_target
import datasets
from datasets import load_dataset, DatasetDict
import logging
import numpy as np
import torch
import re
import random
import os
import json
from promptsource.promptsource.templates import DatasetTemplates

logger = logging.getLogger(__name__)

# TODO : Eval own, Eval on T0
use_verbalizer = True

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "scitail","cos_e"]
    large_data_without_all_splits = ["imdb","amazon_polarity","app_reviews","yelp_review_full","qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
                                     "amazon_polarity",  "newsqa", "searchqa", "triviaqa", "nq", "lama",
                                     "trivia_qa"]

    def __init__(self, config, prompt, seed=42):
        self.config = config
        self.seed = seed
        self.prompt = prompt

    def get_max_target_length(self, tokenizer, default_max_length):
        #if self.labels_list is not None:
        #    return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix]+sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': self.name,
                'extra_fields': extra_fields}

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, script_version="master")

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        # if self.name in ["rotten_tomatoes","imdb","app_reviews"]:
        #     x = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix))            
        # elif self.name[0:16]=='yelp_review_full':
        #     x = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix))
        #     if len(x)== 650000:
        #         indices = self.get_split_indices('train', x, validation_size=100)
        #         x = self.subsample(x, 10000, indices)
        #     elif len(x)==50000:
        #         indices = self.get_split_indices('validation',x,validation_size=100)
        #         x = self.subsample(x, 100, indices)
        # elif self.name[0:9]=='hotpot_qa':
        #     x = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix))
        # if self.name=='wiki_auto':
        #     x = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix))
        #     x.remove_columns('translation')
        
        x = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                        remove_columns=dataset.column_names)
        return x

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, lang=None, file_name=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if self.name in ['wiki_hop','amazon_polarity','yelp_review_full','dbpedia_14','trec','imdb','app_reviews','gigaword','rotten_tomatoes','wiki_auto']:
            print('@@@@@@@@@@@@@@@@@@@@@')
            print('Choice 0')
            print(self.name)
            print('@@@@@@@@@@@@@@@@@@@@@')
            dataset = self.load_dataset(split=split)
            
            print(len(dataset))
            
        elif split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            print('@@@@@@@@@@@@@@@@@@@@@')
            print('Choice 1')
            print('@@@@@@@@@@@@@@@@@@@@@')
            mapped_split = self.split_to_data_split["validation"]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files=file_name, split="train")
            else:
                dataset = self.load_dataset(split=mapped_split)
            #indices = self.get_split_indices(
            #    split, dataset, validation_size=len(dataset)//2)
            indices = self.get_split_indices(
                split, dataset, validation_size=n_obs)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            print('@@@@@@@@@@@@@@@@@@@@@')
            print('Choice 2')
            print('@@@@@@@@@@@@@@@@@@@@@')
            # if self.name[0:9] == 'hotpot_qa':
            #     dataset = self.load_dataset(split='train')
            
            if lang is not None:
                dataset = self.load_dataset(split="train", lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files=file_name, split="train")
            else:
                dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(
                split, dataset, validation_size=n_obs)
            print('^^^^^^^^^^^^^^^^^')
            print('DEBUG 1')
            print(dataset)
            print(len(dataset))
            print('^^^^^^^^^^^^^^^^^')
            dataset = self.subsample(dataset, n_obs, indices)
            #dataset = self.map_dataset(dataset, add_prefix)     
            print('^^^^^^^^^^^^^^^^^')
            print('DEBUG 2')
            print(dataset)
            print(len(dataset))
            print('^^^^^^^^^^^^^^^^^')
            
        else:
            print('@@@@@@@@@@@@@@@@@@@@@')
            print('Choice 3')
            print('@@@@@@@@@@@@@@@@@@@@@')
            mapped_split = self.split_to_data_split[split]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)

            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files=file_name, split="train")
            else:
                dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        print('^^^^^^^^^^^^^^^^^')
        print('DEBUG 3')
        print(len(dataset))
        print(dataset)
        print('^^^^^^^^^^^^^^^^^')
        # if self.name=='1_wiki_auto':
        #     dataset = dataset['train']
        #     print(dataset)
        #     print(dataset[0])
        dataset = self.map_dataset(dataset, add_prefix)     
        assert 'source' not in dataset, "This is the problem!@@@@@@@@@@@@@@@@@"
        print('^^^^^^^^^^^^^^^^^')
        print('DEBUG 4')
        print(len(dataset))
        print(dataset)
        print('^^^^^^^^^^^^^^^^^')
        return dataset
        

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        print('############ SQUAD ##################')
        print(self.name)
        print('####################################')
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)

class HotpotQA(AbstractTask):
    name = "hotpotqa"
    metric = [metrics.squad]

    def load_dataset(self, split):
        print('############ HOTPOTQA ##################')
        print(self.name)
        print('####################################')
        return datasets.load_dataset('hotpot_qa','fullwiki', split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answer']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)

class TriviaQA(AbstractTask):
    name = "triviaqa"
    metric = [metrics.squad]

    def load_dataset(self, split):
        print('############ TRIVIAQA ##################')
        print(self.name)
        print('####################################')
        return datasets.load_dataset('trivia_qa','rc', split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answer']['value']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['search_results'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)

class LAMA(AbstractTask):
    name = "lama"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        print('############ LAMA ##################')
        print(self.name)
        print('####################################')
        print(os.getcwd())
        return datasets.load_dataset("csv",data_files={"eval":'./data/lama_filter.csv'},script_version="master")["eval"]

    def preprocessor(self, example, add_prefix):
        return {'source': "Fill the mask with the missing word. "+example['input'],
                'target': str(example['output']),
                'task': self.name,
                'extra_fields': {}}

class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["premise:", example['sentence1'],
                     "hypothesis:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class Amazon_Polarity(AbstractTask):
    name = "amazon_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MultiNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ANLI(AbstractTask):
    name = "anli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train_r3",
                           "validation": "dev_r3",
                           "test": "test_r3"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        split_to_data_split = {"train": "train_r3",
                               "validation": "dev_r3",
                               "test": "test_r3"}
        return datasets.load_dataset('anli', split=split_to_data_split[split] if split in split_to_data_split else split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


# class SuperGLUERTE(AbstractTask):
#     name = "superglue-rte"
#     labels_list = ['0', '1']
#     split_to_data_split = {"train": "train",
#                            "validation": "validation",
#                            "test": "validation"}
#     metric = [metrics.accuracy]
#     metric_names = ["accuracy"]

#     def load_dataset(self, split):
#         return datasets.load_dataset('super_glue', 'rte', split=split, script_version="master")

#     def preprocessor(self, example, add_prefix=True):
#         src_texts = ["premise:", example["premise"],
#                      "hypothesis:", example["hypothesis"]]
#         tgt_texts = [str(example["label"])]
#         return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['Yes', 'No']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        #tgt_texts = [str(example["label"])]
        tgt_texts = ["Yes" if str(example["label"])=="0" else "No"]

        #return self.seq2seq_format(src_texts, tgt_texts, add_prefix=True, prefix)

        return {'source': "Suppose " + example["premise"] + " Can we infer that " + example["hypothesis"] + "? Yes or no?",
                'target': ' '.join(tgt_texts),
                'task': self.name,
                'extra_fields': {}}

class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'copa', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        
        #src_texts = ["premise:", example["premise"],
        #             "choice1:", example["choice1"],
        #             "choice2:", example["choice2"]]
        src_text = example['premise'] + "This happened because...\n\n Help me pick the more plausible option:\n\n-" + example['choice1'] + "\n-" + example['choice2']
        tgt_text = [example['choice1'] if str(example["label"])=="1" else example['choice2']]
        #tgt_text = [str(example["label"])]
        #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
        return {'source': src_text,
                'target': ' '.join(tgt_text),
                'task': self.name,
                'labels_list': [example['choice1'],example['choice2']],
                'extra_fields': {}}

class COSe(AbstractTask):
    name = "cos_e"
    labels_list = ['0', '1', '2', '3', '4']
    split_to_data_split = {"train": "train",
                           "validation": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('cos_e', 'v1.11', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_text = example['question'] + "\n\n- " + example['choices'][0] + "\n- " + example['choices'][1] + "\n- " + example['choices'][2] + "\n- " + example['choices'][3] + "\n- " + example['choices'][4] + "\n\n" + "The best answer is " 
        
        #src_texts = ["premise:", example["premise"],
        #             "choice1:", example["choice1"],
        #             "choice2:", example["choice2"]]
        tgt_text = [str(example["answer"])]
        #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
        return {'source': src_text,
                'target': ' '.join(tgt_text),
                'task': self.name,
                'extra_fields': {}}


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers,
              metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1", "em"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'multirc', split=split, script_version="master")

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, add_prefix=True):
        group = example['idx']['question']
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields={"group": group})


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wic', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wsc.fixed', split=split, script_version="master")

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'record', split=split, script_version="master")

    def preprocessor(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend(
                [{"answers": ex["answers"]}]*num_duplicates)
        return new_batch

    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           batched=True, remove_columns=dataset.column_names)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', "winogrande_xl", split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"],
                     "option0:", example["option1"],
                     "option1:", example["option1"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


# class PAWS(AbstractTask):
#     name = "paws"
#     labels_list = ["0", "1"]
#     metric = [metrics.accuracy]
#     metric_names = ["accuracy"]
#     split_to_data_split = {"train": "train",
#                            "validation": "validation",
#                            "test": "test"}

#     def load_dataset(self, split):
#         return datasets.load_dataset('paws', 'labeled_final', split=split, script_version="master")

#     def preprocessor(self, example, add_prefix=True):
#         src_texts = ["sentence1:", example['sentence1'],
#                      "sentence2:", example["sentence2"]]
#         tgt_texts = [str(example['label'])]
#         return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["No", "Yes"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split: int):
        return datasets.load_dataset('paws', 'labeled_final', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"], "sentence2:", example["sentence2"]]
        #tgt_texts = [str(example["label"])]
        tgt_texts = ["No" if str(example["label"])=="0" else "Yes"]

        #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
        return {'source': example["sentence1"]+"\nIs that a paraphrase of the following sentence?\n"+example['sentence2']+"?\n",
                'target': ' '.join(tgt_texts),
                'task': self.name,
                'extra_fields': {}}

class WIKIHOP(AbstractTask):
    name = "wiki_hop"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        wiki_hop = load_dataset('wiki_hop')
        wiki_hop_train = wiki_hop['train'].select([i for i in range(0,50000)])
        wiki_hop_eval = wiki_hop['validation'].select([i for i in range(0,300)])
        wiki_hop = DatasetDict()
        wiki_hop['train'] = wiki_hop_train
        wiki_hop['validation'] = wiki_hop_eval
        if split=='train':
            return wiki_hop['train']
        elif split=='validation':
            return wiki_hop['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('wiki_hop/original')[self.prompt]
        result = prompt.apply(example)
        source = result[0]
        target = result[1]
        
        if use_verbalizer:
            if self.prompt in ['choose_best_object_interrogative_1','choose_best_object_affirmative_1','choose_best_object_affirmative_3','choose_best_object_affirmative_2','choose_best_object_interrogative_2']:
                return {'source': source,
                        'target': target,
                        'task': self.name,
                        'labels_list' : [ex for ex in example['candidates']],
                        'extra_fields': {}}
        return {'source': source,
                'target': target,
                'task': self.name,
                'extra_fields': {}}
class AMAZONPOLARITY(AbstractTask):
    name = "amazon_polarity"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        train_data_files = {"train":"./data/manual/amazon_review_polarity_csv/train.csv"}

        amazon_polarity = load_dataset("csv",data_files=train_data_files, script_version="master",names=['label','title','content'])
        
        amazon_polarity_train = amazon_polarity['train'].select([i for i in range(0,50000)])
        amazon_polarity_eval = amazon_polarity['train'].select([len(amazon_polarity_train)-i for i in range(0,300)])
        
        amazon_polarity = DatasetDict()
        amazon_polarity['train'] = amazon_polarity_train
        amazon_polarity['validation'] = amazon_polarity_eval
        if split=='train':
            return amazon_polarity['train']
        elif split=='validation':
            return amazon_polarity['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('amazon_polarity')[self.prompt]
        result = prompt.apply(example)
        if self.prompt == "Is_this_review":
            #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'Negative' if example['label']==1 else 'Positive',
                        'labels_list': ['Positive','Negative'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'Negative' if example['label']==1 else 'Positive',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "User_recommend_this_product":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'No' if example['label']==1 else 'Yes',
                        'labels_list': ['Yes','No'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'No' if example['label']==1 else 'Yes',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "Is_this_product_review_positive":
            if use_verbalizer:
                return {'source': result[0],
                    'target': 'No' if example['label']==1 else 'Yes',
                    'labels_list': ['Yes','No'],
                    'task': self.name,
                    'extra_fields': {}}
            return {'source': result[0],
                    'target': 'No' if example['label']==1 else 'Yes',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "Is_this_review_negative":
            if use_verbalizer:
                return {'source': result[0],
                    'target': 'Yes' if example['label']==1 else 'No',
                    'labels_list':['No','Yes'],
                    'task': self.name,
                    'extra_fields': {}}
            return {'source': result[0],
                    'target': 'Yes' if example['label']==1 else 'No',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "convey_negative_or_positive_sentiment":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'Negative' if example['label']==1 else 'Positive',
                        'labels_list':['Positive','Negative'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'Negative' if example['label']==1 else 'Positive',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "negative_or_positive_tone":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'Negative' if example['label']==1 else 'Positive',
                        'labels_list':['Positive','Negative'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'Negative' if example['label']==1 else 'Positive',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "user_satisfied":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'dissatisfied' if example['label']==1 else 'satisfied',
                        'labels_list':['satisfied','dissatisfied'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'dissatisfied' if example['label']==1 else 'satisfied',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "would_you_buy":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'decrease' if example['label']==1 else 'increase',
                        'labels_list':['increase','decrease'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'decrease' if example['label']==1 else 'increase',
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt =="flattering_or_not":
            if use_verbalizer:
                return {'source': result[0],
                        'target': 'unflattering' if example['label']==1 else 'flattering',
                        'labels_list':['flattering','unfalttering'],
                        'task': self.name,
                        'extra_fields': {}}
            return {'source': result[0],
                    'target': 'unflattering' if example['label']==1 else 'flattering',
                    'task': self.name,
                    'extra_fields': {}}
class YELPREVIEWFULL(AbstractTask):
    name = "yelp_review_full"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        train_data_files = {"train":"./data/manual/yelp_review_full_csv/train.csv"}
        eval_data_files = {"validation":"./data/manual/yelp_review_full_csv/test.csv"}

        yelp_review_train = load_dataset("csv",data_files=train_data_files, script_version="master",names=['label','text'])
        yelp_review_eval = load_dataset("csv",data_files=eval_data_files, script_version="master",names=['label','text'])
        yelp_review_train = yelp_review_train['train'].select([i for i in range(0,50000)])
        yelp_review_eval = yelp_review_eval['validation'].select([i for i in range(1,301)])
        yelp_review = DatasetDict()
        yelp_review['train'] = yelp_review_train
        yelp_review['validation'] = yelp_review_eval
        if split=='train':
            return yelp_review['train']
        elif split=='validation':
            return yelp_review['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('yelp_review_full')[self.prompt]
        result = prompt.apply(example)
        if self.prompt=='format_score':
            if use_verbalizer:
                return {'source': result[0],
                        'target': str(example['label']),
                        'labels_list':['1','2','3','4','5'],
                        'task': self.name,
                        'extra_fields': {}}
            else:
                return {'source': result[0],
                        'target': str(example['label']),
                        'task': self.name,
                        'extra_fields': {}}
        else:
            #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
            if use_verbalizer:
                return {'source': result[0],
                    'target': "1 star" if example['label']==1 else str(example['label'])+" stars",
                    'labels_list':['1 star','2 stars','3 stars','4 stars','5 stars'],
                    'task': self.name,
                    'extra_fields': {}}
            else:
                return {'source': result[0],
                        'target': "1 star" if example['label']==1 else str(example['label'])+" stars",
                        'task': self.name,
                        'extra_fields': {}}
class DBPEDIA14(AbstractTask):
    name = "dbpedia_14"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        train_data_files = {"train":"./data/manual/dbpedia_csv/train.csv"}
        eval_data_files = {"validation":"./data/manual/dbpedia_csv/test.csv"}

        dbpedia_train = load_dataset("csv",data_files=train_data_files, script_version="master",names=['label','title','content'])
        dbpedia_eval = load_dataset("csv",data_files=eval_data_files, script_version="master",names=['label','title','content'])
        dbpedia = DatasetDict()
        dbpedia['train'] = dbpedia_train['train'].select([i for i in range(0,50000)])
        dbpedia['validation'] = dbpedia_eval['validation'].select([i for i in range(0,300)])
        
        if split=='train':
            return dbpedia['train']
        elif split=='validation':
            return dbpedia['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('dbpedia_14')[self.prompt]
        results = prompt.apply(example)
        
        result = {'source': results[0],
                'target': results[1],
                'task': self.name,
                'extra_fields': {}}
        if use_verbalizer:
            if self.prompt == "given_list_what_category_does_the_paragraph_belong_to":
                result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
            elif self.prompt == "pick_one_category_for_the_following_text":
                result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
            elif self.prompt == "given_a_choice_of_categories":
                result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
            elif self.prompt == "given_a_list_of_category_what_does_the_title_belong_to":
                result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
        return result
class TREC(AbstractTask):
    name = "trec"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        trec = load_dataset('trec',version='2.0.0')
        trec_train = trec['train']
        trec_validation = trec['test']
        if 'label-coarse' in trec_train[0]:
            trec_train = trec_train.rename_column('label-coarse','label_coarse')
        elif 'coarse_label' in trec_train[0]:
            trec_train = trec_train.rename_column('coarse_label','label_coarse')
        if 'label-fine' in trec_train[0]:
            trec_train = trec_train.rename_column('label-fine','label_fine')
        elif 'fine_label' in trec_train[0]:
            trec_train = trec_train.rename_column('fine_label','label_fine')
        if 'label-coarse' in trec_validation[0]:
            trec_validation = trec_validation.rename_column('label-coarse','label_coarse')
        elif 'coarse_label' in trec_validation[0]:
            trec_validation = trec_validation.rename_column('coarse_label','label_coarse')
        if 'label-fine' in trec_validation[0]:
            trec_validation = trec_validation.rename_column('label-fine','label_fine')
        elif 'fine_label' in trec_validation[0]:
            trec_validation = trec_validation.rename_column('fine_label','label_fine')
        
        trec = DatasetDict()
        trec['train'] = trec_train
        trec['validation'] = trec_validation
        print('################')
        print(trec['train'][0])
        print('################')
        if self.prompt == "fine_grained_LOC":
            
            trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==5)
            trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==5)
            # except:
            #     trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==5)
            #     trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==5)
        elif self.prompt == "fine_grained_NUM_context_first":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==4)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==4)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==4)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==4)
        elif self.prompt == "fine_grained_ENTY":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==1)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==1)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==1)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==1)
        elif self.prompt == "fine_grained_NUM":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==4)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==4)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==4)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==4)
        elif self.prompt == "fine_grained_LOC_context_first":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==5)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==5)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==5)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==5)
        elif self.prompt == "fine_grained_DESC":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==0)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==0)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==0)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==0)
        elif self.prompt == "fine_grained_ABBR":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==2)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==2)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==2)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==2)
        elif self.prompt == "fine_grained_ABBR_context_first":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==2)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==2)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==2)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==2)
        elif self.prompt == "fine_grained_HUM":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==3)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==3)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==3)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==3)
        elif self.prompt == "fine_grained_HUM_context_first":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==3)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==3)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==3)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==3)
        elif self.prompt == "fine_grained_DESC_context_first":
            try:
                trec['train'] = trec['train'].filter(lambda ex: ex['label_coarse']==0)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label_coarse']==0)
            except:
                trec['train'] = trec['train'].filter(lambda ex: ex['label-coarse']==0)
                trec['validation'] = trec['validation'].filter(lambda ex: ex['label-coarse']==0)
        
        if split=='train':
            return trec['train']
        elif split=='validation':
            return trec['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('trec')[self.prompt]
        results = prompt.apply(example)
        
        result = {'source': results[0],
                'target': results[1],
                'task': self.name,
                'extra_fields': {}}
        if use_verbalizer:
            if self.prompt == "what_category_best_describe":
                result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
            elif self.prompt == "fine_grained_LOC":
                result['labels_list'] = ["city","country","mountain","state","other location"]
            elif self.prompt == "fine_grained_NUM_context_first":
                result['labels_list'] = ["code","count","date","distance","price","order","period of time","percentage","speed","temperature","size","weight","other number"]
            elif self.prompt == "fine_grained_ENTY":
                result['labels_list'] = ["an animal","an organ of the body","a color","creative piece","currency","disease or medicine","event","food","musical instrument","language","letter","plant","product","religion","sport","substance","symbol","technique","term","vehicle","word","other entity"]
            elif self.prompt == "fine_grained_NUM":
                result['labels_list'] = ["code","count","date","distance","price","order","period of time","percentage","speed","temperature","size","weight","other number"]
            elif self.prompt == "pick_the_best_descriptor":
                result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
            elif self.prompt == "fine_grained_open_context_first":
                result['labels_list'] = ["Manner","Creative Piece","Animal","Expression abbreviated","Individual","Group","Title","Defintion","Date","Reason","Event","State","Description","Count","Other","Letter","Religion","Food","Country","Color","Term","City","Organ of the body","Disease or medicine","Mountain","Price","Product","Period","Substance","Sport","Plant","Technique","Size","Instrument","Abbreviation","Speed","Word","Language","Percentage","Code","Distance","Temperature","Symbol","Order","Vehicle","Weight","Currency"]
            elif self.prompt == "fine_grained_LOC_context_first":
                result['labels_list'] = ["city","country","mountain","state","other location"]
            elif self.prompt == "which_category_best_describes":
                result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
            elif self.prompt == "fine_grained_DESC":
                result['labels_list'] = ["definition","description","manner of action","reason"]
            elif self.prompt == "trec1":
                result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
            elif self.prompt == "fine_grained_ABBR":
                result['labels_list'] = ["abbreviation","expression abbreviated"]
            elif self.prompt == "fine_grained_ABBR_context_first":
                result['labels_list'] = ["abbreviation","expression abbreviated"]
            elif self.prompt == "trec2":
                result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
            elif self.prompt == "fine_grained_HUM":
                result['labels_list'] = ["group","individual","title","description"]
            elif self.prompt == "fine_grained_open":
                result['labels_list'] = ["Manner","Creative Piece","Animal","Expression abbreviated","Individual","Group","Title","Defintion","Date","Reason","Event","State","Description","Count","Other","Letter","Religion","Food","Country","Color","Term","City","Organ of the body","Disease or medicine","Mountain","Price","Product","Period","Substance","Sport","Plant","Technique","Size","Instrument","Abbreviation","Speed","Word","Language","Percentage","Code","Distance","Temperature","Symbol","Order","Vehicle","Weight","Currency"]
            elif self.prompt == "fine_grained_HUM_context_first":
                result['labels_list'] = ["group","individual","title","description"]
            elif self.prompt == "fine_grained_DESC_context_first":
                result['labels_list'] = ["definition","description","manner of action","reason"]
        return result
class IMDB(AbstractTask):
    name = "imdb"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        imdb = load_dataset('imdb')
        imdb['validation'] = imdb['test']
        imdb_train = imdb['train'].select([i for i in range(0,50000)])
        imdb_validation = imdb['validation'].select([i for i in range(0,300)])
        imdb = DatasetDict()
        imdb['train'] = imdb_train
        imdb['validation'] = imdb_validation
        
        if split=='train':
            return imdb['train']
        elif split=='validation':
            return imdb['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('imdb')[self.prompt]
        results = prompt.apply(example)
        
        result= {'source': results[0],
                'target': results[1],
                'task': self.name,
                'extra_fields': {}}
        if use_verbalizer:
            if self.prompt == "Movie Expressed Sentiment 2":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Opinion bad good choices":
                result['labels_list'] = ["bad","good"]
            elif self.prompt == "Sentiment with choices ":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Sentiment Feeling":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Writer Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Movie Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Text Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Negation template for positive and negative":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Enjoyment Yes No":
                result['labels_list'] = ["No","Yes"]
            elif self.prompt == "Reviewer Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Enjoyment":
                result['labels_list'] = ["They didn't like it!","They loved it"]
        return result
class APPREVIEWS(AbstractTask):
    name = "app_reviews"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        app_reviews = load_dataset('app_reviews')

        eval_app_reviews = app_reviews['train'].select([len(app_reviews['train'])-i for i in range(0,300)])
        train_app_reviews = app_reviews['train'].select([i for i in range(0,50000)])
        app_reviews = DatasetDict()
        app_reviews['train'] = train_app_reviews
        app_reviews['validation'] = eval_app_reviews
        
        if split=='train':
            return app_reviews['train']
        elif split=='validation':
            return app_reviews['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('app_reviews')[self.prompt]
        results = prompt.apply(example)
        
        result= {'source': results[0],
                'target': results[1],
                'task': self.name,
                'extra_fields': {}}
        if use_verbalizer:
            if self.prompt == "categorize_rating_using_review":
                result['labels_list'] = ["Not at all","No","Maybe","Yes","Definitely"]
            elif self.prompt == "convert_to_star_rating":
                result['labels_list'] = ["\u2605","\u2605\u2605","\u2605\u2605\u2605","\u2605\u2605\u2605\u2605","\u2605\u2605\u2605\u2605\u2605"]
        return result
class GIGAWORD(AbstractTask):
    name = "gigaword"
    metric = [metrics.calculate_rouge,metrics.bleu]
    metric_names = ["rouge","bleu"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        gigaword = load_dataset('gigaword')
        gigaword_train = gigaword['train'].select([i for i in range(0,10000)])
        gigaword_eval = gigaword['validation'].select([i for i in range(0,300)])
        gigaword = DatasetDict()
        gigaword['train'] = gigaword_train
        gigaword['validation'] = gigaword_eval
        
        if split=='train':
            return gigaword['train']
        elif split=='validation':
            return gigaword['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('gigaword')[self.prompt]
        result = prompt.apply(example)
        
        return {'source': result[0],
                'target': result[1],
                'task': self.name,
                'extra_fields': {}}
class MULTINEWS(AbstractTask):
    name = "multi_news"
    metric = [metrics.calculate_rouge,metrics.bleu]
    metric_names = ["rouge","bleu"]
    split_to_data_split = {"train":"train","validation": "validation"}
    

    def load_dataset(self, split: int):
        train_src_dir = './data/manual/multi_news/train.src.cleaned'
        train_tgt_dir = './data/manual/multi_news/train.tgt'
        eval_src_dir = './data/manual/multi_news/val.src.cleaned'
        eval_tgt_dir = './data/manual/multi_news/val.tgt'
        def _generate_examples(src_file, tgt_file):
            result = {"document":[],"summary":[]}
            with open(src_file, encoding="utf-8") as src_f, open(tgt_file, encoding="utf-8") as tgt_f:
                for i, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                    
                    # In original file, each line has one example and natural newline
                    # tokens "\n" are being replaced with "NEWLINE_CHAR". Here restore
                    # the natural newline token to avoid special vocab "NEWLINE_CHAR".
                    result['document'].append(src_line.strip().replace("NEWLINE_CHAR", "\n"))
                    result["summary"].append(tgt_line.strip())
            return result
        multi_news_train = datasets.Dataset.from_dict(_generate_examples(train_src_dir,train_tgt_dir)).select([i for i in range(0,10000)])
        multi_news_eval = datasets.Dataset.from_dict(_generate_examples(eval_src_dir,eval_tgt_dir)).select([i for i in range(0,300)])
        multi_news = DatasetDict()
        multi_news['train'] = multi_news_train
        multi_news['validation'] = multi_news_eval
        
        if split=='train':
            return multi_news['train']
        elif split=='validation':
            return multi_news['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        if example['document']=='':
            example['document'] = example['summary']
        prompt = DatasetTemplates('multi_news')[self.prompt]
        result = prompt.apply(example)
    
        return {'source': result[0],
                'target': result[1],
                'task': self.name,
                'extra_fields': {}}
        
class ROTTENTOMATOES(AbstractTask):
    name = "rotten_tomatoes"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        rotten_tomatoes = load_dataset('rotten_tomatoes')
        rotten_tomatoes_train = rotten_tomatoes['train']
        rotten_tomatoes_eval = rotten_tomatoes['validation'].select([i for i in range(0,300)])
        rotten_tomatoes = DatasetDict()
        rotten_tomatoes['train'] = rotten_tomatoes_train
        rotten_tomatoes['validation'] = rotten_tomatoes_eval
        
        if split=='train':
            return rotten_tomatoes['train']
        elif split=='validation':
            return rotten_tomatoes['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        prompt = DatasetTemplates('rotten_tomatoes')[self.prompt]
        results = prompt.apply(example)
        
        result= {'source': results[0],
                'target': results[1],
                'task': self.name,
                'extra_fields': {}}                                                
        if use_verbalizer:
            if self.prompt == "Reviewer Opinion bad good choices":
                result['labels_list'] = ["bad","good"]
            elif self.prompt == "Text Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Sentiment with choices ":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Enjoyment Yes No":
                result['labels_list'] = ["No","Yes"]
            elif self.prompt == "Reviewer Enjoyment":
                result['labels_list'] = ["They didn't like it","They loved it"]
            elif self.prompt == "Movie Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Writer Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Movie Expressed Sentiment 2":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Expressed Sentiment":
                result['labels_list'] = ["negative","positive"]
            elif self.prompt == "Reviewer Sentiment Feeling":
                result['labels_list'] = ["negative","positive"]
        return result
class STORYCLOZE(AbstractTask):
    name = "story_cloze"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"validation": "validation"}

    def load_dataset(self, split: int):
        import os
        print('#####')
        print(os.getcwd())
        print('#####')
        x = datasets.load_dataset('csv', data_files='./data/manual/2016.csv', script_version="master")
        x['validation'] = x['train']
        return x['validation']

    def preprocessor(self, example, add_prefix=True):
        if self.prompt == "Generate Ending":
            #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
            return {'source': "Generate a possible ending for the following story: "+example['InputSentence1']+ " "+example['InputSentence2']+" "+example['InputSentence3']+" "+example['InputSentence4']+" ",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt == "Choose Story Ending":
            return {'source': "Read the following story :\n\n"+example['InputSentence1']+ "\n"+example['InputSentence2']+"\n"+example['InputSentence3']+"\n"+example['InputSentence4']+"\n\nChoose a possible ending for the previous story from the following options: \n- "+example['RandomFifthSentenceQuiz1']+"\n -"+example['RandomFifthSentenceQuiz2']+"\n",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'labels_list': [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']],
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt=='Answer Given options':
            return {'source': example['InputSentence1']+ " "+example['InputSentence2']+" "+example['InputSentence3']+" "+example['InputSentence4']+" What is a possible continuation for the story given the following options ? \n- "+example['RandomFifthSentenceQuiz1']+"\n- "+example['RandomFifthSentenceQuiz2']+"\n",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'labels_list': [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']],
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt=='Movie What Happens Next':
            return {'source': "Yesterday, I watched a movie. Here's what happened: "+example['InputSentence1']+ " "+example['InputSentence2']+" "+example['InputSentence3']+" "+example['InputSentence4']+" What happens next? - "+example['RandomFifthSentenceQuiz1']+"\n- "+example['RandomFifthSentenceQuiz2']+"\n",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'labels_list': [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']],
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt=='Story Continuation and Options':
            return {'source': "What is a possible continuation for the following story ? \n\n"+example['InputSentence1']+ "\n"+example['InputSentence2']+"\n"+example['InputSentence3']+"\n"+example['InputSentence4']+"\n\nChoose from the following options:\n- "+example['RandomFifthSentenceQuiz1']+"\n- "+example['RandomFifthSentenceQuiz2']+"\n",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'labels_list': [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']],
                    'task': self.name,
                    'extra_fields': {}}
        elif self.prompt=='Novel Correct Ending':
            return {'source': "I read the following novel: "+example['InputSentence1']+ " "+example['InputSentence2']+" "+example['InputSentence3']+" "+example['InputSentence4']+" "+"What do you think is the most probable ending? You can choose from the following options:\n- "+example['RandomFifthSentenceQuiz1']+"\n- "+example['RandomFifthSentenceQuiz2']+"\n",
                    'target': example['RandomFifthSentenceQuiz1'] if example['AnswerRightEnding']==1 else example['RandomFifthSentenceQuiz2'],
                    'labels_list': [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']],
                    'task': self.name,
                    'extra_fields': {}}
class ANLIR1(AbstractTask):
    name = "anli_dev_r1"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"validation": "validation"}
    def __init__(self, config, prompt, seed=42):
        self.config = config
        self.seed = seed
        self.prompt = prompt
        self.promptList = DatasetTemplates("anli")[self.prompt]

    def load_dataset(self, split: int):
        x = datasets.load_dataset('anli')
        return x['dev_r1']

    def preprocessor(self, example, add_prefix=True):
        tmp = self.promptList.apply(example)

        src_texts = tmp[:-1]
        tgt_texts = tmp[-1]

        #result = self.seq2seq_format(src_texts, tgt_texts, add_prefix=False)
        result = {}
        result['source'] = ' '.join(src_texts)
        result['target'] = tgt_texts
        result['task'] = self.name
        result['extra_fields'] = {}
        if self.prompt == "can we infer":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "claim true/false/inconclusive":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "MNLI crowdsource":
            result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
        elif self.prompt == "should assume":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "does it follow that":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "GPT-3 style":
            result['labels_list'] = ["True","False","Neither"]
        elif self.prompt == "based on the previous passage":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "justified in saying":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "take the following as truth":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "must be true":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "guaranteed/possible/impossible":
            result['labels_list'] = ["Guaranteed","Possible","Impossible"]
        elif self.prompt == "always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "does this imply":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "consider always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "guaranteed true":
            result['labels_list'] = ["Yes","Maybe","No"]
        return result
class ANLIR2(AbstractTask):
    name = "anli_dev_r2"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"validation": "validation"}
    def __init__(self, config, prompt, seed=42):
        self.config = config
        self.seed = seed
        self.prompt = prompt
        self.promptList = DatasetTemplates("anli")[self.prompt]

    def load_dataset(self, split: int):
        x = datasets.load_dataset('anli')
        return x['dev_r2']

    def preprocessor(self, example, add_prefix=True):
        tmp = self.promptList.apply(example)

        src_texts = tmp[:-1]
        tgt_texts = tmp[-1]

        #result = self.seq2seq_format(src_texts, tgt_texts, add_prefix=False)
        result = {}
        result['source'] = ' '.join(src_texts)
        result['target'] = tgt_texts
        result['task'] = self.name
        result['extra_fields'] = {}
        if self.prompt == "can we infer":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "claim true/false/inconclusive":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "MNLI crowdsource":
            result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
        elif self.prompt == "should assume":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "does it follow that":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "GPT-3 style":
            result['labels_list'] = ["True","False","Neither"]
        elif self.prompt == "based on the previous passage":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "justified in saying":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "take the following as truth":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "must be true":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "guaranteed/possible/impossible":
            result['labels_list'] = ["Guaranteed","Possible","Impossible"]
        elif self.prompt == "always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "does this imply":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "consider always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "guaranteed true":
            result['labels_list'] = ["Yes","Maybe","No"]
        return result
class ANLIR3(AbstractTask):
    name = "anli_dev_r3"
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"validation": "validation"}
    def __init__(self, config, prompt, seed=42):
        self.config = config
        self.seed = seed
        self.prompt = prompt
        self.promptList = DatasetTemplates("anli")[self.prompt]

    def load_dataset(self, split: int):
        x = datasets.load_dataset('anli')
        return x['dev_r3']

    def preprocessor(self, example, add_prefix=True):
        tmp = self.promptList.apply(example)

        src_texts = tmp[:-1]
        tgt_texts = tmp[-1]

        #result = self.seq2seq_format(src_texts, tgt_texts, add_prefix=False)
        result = {}
        result['source'] = ' '.join(src_texts)
        result['target'] = tgt_texts
        result['task'] = self.name
        result['extra_fields'] = {}
        if self.prompt == "can we infer":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "claim true/false/inconclusive":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "MNLI crowdsource":
            result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
        elif self.prompt == "should assume":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "does it follow that":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "GPT-3 style":
            result['labels_list'] = ["True","False","Neither"]
        elif self.prompt == "based on the previous passage":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "justified in saying":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "take the following as truth":
            result['labels_list'] = ["True","Inconclusive","False"]
        elif self.prompt == "must be true":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "guaranteed/possible/impossible":
            result['labels_list'] = ["Guaranteed","Possible","Impossible"]
        elif self.prompt == "always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "does this imply":
            result['labels_list'] = ["Yes","Maybe","No"]
        elif self.prompt == "consider always/sometimes/never":
            result['labels_list'] = ["Always","Sometimes","Never"]
        elif self.prompt == "guaranteed true":
            result['labels_list'] = ["Yes","Maybe","No"]
        return result
class WIKIAUTO(AbstractTask):
    name = "wiki_auto"
    metric = [metrics.bleu]
    metric_names = ["bleu"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        wiki_auto = DatasetDict()
        with open("./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            wiki_auto['train'] = train_data
        if self.prompt == "simplification_1":
            with open("./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                eval_data = eval_data.select([i for i in range(0,300)])
                wiki_auto['validation'] = eval_data
        elif self.prompt == "simplification_2":
            with open("./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json",'r') as f:
                eval_data2 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data2)
                eval_data = eval_data.select([i for i in range(0,300)])
                wiki_auto['validation'] = eval_data
        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])
        
        
        if split=='train':
            return wiki_auto['train']
        elif split=='validation':
            return wiki_auto['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt']),
                'task': self.name,
                'extra_fields': {}}

class ASSET(AbstractTask):
    name = "asset"
    metric = [metrics.bleu]
    metric_names = ["bleu"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        if self.prompt == "simplification_1":
            with open("./data/manual/ct0_data/asset/asset/simplification_1.validation.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
        elif self.prompt == "simplification_2":
            with open("./data/manual/ct0_data/asset/asset/simplification_2.validation.json",'r') as f:
                eval_data2 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data2)

        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])

        asset = DatasetDict()
        #asset['train'] = train_data
        asset['validation'] = eval_data
        
        # if split=='train':
        #     return wiki_auto['train']
        if split=='validation':
            return asset['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt'][0]),
                'task': self.name,
                'extra_fields': {}}

class HAIKU(AbstractTask):
    name = "haiku"
    metric = [metrics.calculate_rouge]
    metric_names = ["rouge"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        haiku = DatasetDict()
        with open("./data/manual/ct0_data/haiku/haiku/do_nothing.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            haiku['train'] = train_data
        with open("./data/manual/ct0_data/haiku/haiku/do_nothing.test.json",'r') as f:
            eval_data1 = json.load(f)
            eval_data = datasets.Dataset.from_dict(eval_data1)
            haiku['validation'] = eval_data
        
        if split=='train':
            return haiku['train']
        elif split=='validation':
            return haiku['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': example['src'],
                'target': example['tgt'],
                'task': self.name,
                'extra_fields': {}}

class CT0_GIGAWORD(AbstractTask):
    name = "ct0_gigaword"
    metric = [metrics.calculate_rouge]
    metric_names = ["rouge"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        ct0_gigaword = DatasetDict()
        if self.prompt=="constrain_contain+make_a_title":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_contain+make_a_title.train.json",'r') as f:
                train_data = json.load(f)
                train_data = datasets.Dataset.from_dict(train_data)
                ct0_gigaword['train'] = train_data
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_contain+make_a_title.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        elif self.prompt=="constrain_contain+write_its_sentence":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_contain+write_its_sentence.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        elif self.prompt=="constrain_end+make_a_title":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_end+make_a_title.train.json",'r') as f:
                train_data = json.load(f)
                train_data = datasets.Dataset.from_dict(train_data)
                ct0_gigaword['train'] = train_data
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_end+make_a_title.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        elif self.prompt=="constrain_end+write_its_sentence":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_end+write_its_sentence.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        elif self.prompt=="constrain_start+make_a_title":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_start+make_a_title.train.json",'r') as f:
                train_data = json.load(f)
                train_data = datasets.Dataset.from_dict(train_data)
                ct0_gigaword['train'] = train_data
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_start+make_a_title.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        elif self.prompt=="constrain_start+write_its_sentence":
            with open("./data/manual/ct0_data/gigaword/gigaword/constrain_start+write_its_sentence.test.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                ct0_gigaword['validation'] = eval_data
        
        if split=='train':
            return ct0_gigaword['train']
        elif split=='validation':
            return ct0_gigaword['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': example['src'],
                'target': example['tgt'],
                'task': self.name,
                'extra_fields': {}}

class COVIDQA(AbstractTask):
    name = "covid_qa"
    metric = [metrics.bertscore]
    metric_names = ["bertscore"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        covid_qa = DatasetDict()
        with open("./data/manual/ct0_data/covid_qa/covid_qa_deepset/covid_cloze_book_qa.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            covid_qa['train'] = train_data
            covid_qa['validation'] = train_data
        
        if split=='train':
            return covid_qa['train']
        elif split=='validation':
            return covid_qa['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': example['src'],
                'target': example['tgt'],
                'task': self.name,
                'extra_fields': {}}
    
class ELI5(AbstractTask):
    name = "eli5"
    metric = [metrics.bertscore]
    metric_names = ["bertscore"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        eli5 = DatasetDict()
        if self.prompt == "generate_a_question_1":
            with open("./data/manual/ct0_data/eli5/generate_a_question_1.train_asks.json",'r') as f:
                train_data = json.load(f)
                train_data = datasets.Dataset.from_dict(train_data)
                eli5['train'] = train_data
            with open("./data/manual/ct0_data/eli5/generate_a_question_1.test_asks.json",'r') as f:
                eval_data1 = json.load(f)
                eval_data = datasets.Dataset.from_dict(eval_data1)
                eval_data = eval_data.select([i for i in range(0,300)])
                eli5['validation'] = eval_data
        # elif self.prompt == "generate_a_question_2":
        #     with open("./data/manual/ct0_data/eli5/generate_a_question_2.test_asks.json",'r') as f:
        #         eval_data2 = json.load(f)
        #         eval_data = datasets.Dataset.from_dict(eval_data2)
        #         eval_data = eval_data.select([i for i in range(0,300)])
        #         eli5['validation'] = eval_data
        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])
        
        
        if split=='train':
            return eli5['train']
        elif split=='validation':
            return eli5['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt']),
                'task': self.name,
                'extra_fields': {}}
class EMDG(AbstractTask):
    name = "emdg"
    metric = [metrics.bertscore]
    metric_names = ["bertscore"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        emdg = DatasetDict()
        
        with open("./data/manual/ct0_data/empathetic_dialogues/empathetic_dialogues/dialogue_with_emotion.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            emdg['train'] = train_data
        with open("./data/manual/ct0_data/empathetic_dialogues/empathetic_dialogues/dialogue_with_emotion.test.json",'r') as f:
            eval_data1 = json.load(f)
            eval_data = datasets.Dataset.from_dict(eval_data1)
            eval_data = eval_data.select([i for i in range(0,300)])
            emdg['validation'] = eval_data
        
        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])
        
        
        if split=='train':
            return emdg['train']
        elif split=='validation':
            return emdg['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt']),
                'task': self.name,
                'extra_fields': {}}

class eSNLI(AbstractTask):
    name = "esnli"
    metric = [metrics.bertscore]
    metric_names = ["bertscore"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        esnli = DatasetDict()
        
        with open("./data/manual/ct0_data/eSNLI/eSNLI/explain_why.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            esnli['train'] = train_data
        with open("./data/manual/ct0_data/eSNLI/eSNLI/explain_why.test.json",'r') as f:
            eval_data1 = json.load(f)
            eval_data = datasets.Dataset.from_dict(eval_data1)
            eval_data = eval_data.select([i for i in range(0,300)])
            esnli['validation'] = eval_data
        
        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])
        
        
        if split=='train':
            return esnli['train']
        elif split=='validation':
            return esnli['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt']),
                'task': self.name,
                'extra_fields': {}}

class TWITTER(AbstractTask):
    name = "twitter"
    metric = [metrics.bertscore]
    metric_names = ["bertscore"]
    split_to_data_split = {"train":"train","validation": "validation"}

    def load_dataset(self, split: int):
        twitter = DatasetDict()
        
        with open("./data/manual/ct0_data/twitter_top20/twitter_top20/tweet_as+about.train.json",'r') as f:
            train_data = json.load(f)
            train_data = datasets.Dataset.from_dict(train_data)
            twitter['train'] = train_data
        with open("./data/manual/ct0_data/twitter_top20/twitter_top20/tweet_as+about.test.json",'r') as f:
            eval_data1 = json.load(f)
            eval_data = datasets.Dataset.from_dict(eval_data1)
            eval_data = eval_data.select([i for i in range(0,300)])
            twitter['validation'] = eval_data
        
        # wiki_auto_training = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.train.json")['train']
        # wiki_auto_eval1 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_1.test.json")['train']
        # wiki_auto_eval2 = load_dataset("json",data_files="./data/manual/ct0_data/wiki_auto/wiki_auto/simplification_2.test.json")['train']
        #eval_data = datasets.concatenate_datasets([eval_data1, eval_data2])
        
        
        if split=='train':
            return twitter['train']
        elif split=='validation':
            return twitter['validation']
        else:
            return None

    def preprocessor(self, example, add_prefix=True):
        
        return {'source': str(example['src']),
                'target': str(example['tgt']),
                'task': self.name,
                'extra_fields': {}}

TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('cos_e',COSe),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multi_nli', MultiNLI),
        ('snli', SNLI),
        ('newsqa', Squad),
        ('searchqa', Squad),
        ('triviaqa', TriviaQA),
        ('nq', Squad),
        ('hotpotqa', HotpotQA),
        ('lama', LAMA),
        ('anli', ANLI),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp_polarity', YelpPolarity),
        ('paws', PAWS),
        ('story_cloze',STORYCLOZE),
        ('anli_dev_r1', ANLIR1),
        ('anli_dev_r2', ANLIR2),
        ('anli_dev_r3', ANLIR3),
        ('amazon_polarity', AMAZONPOLARITY),
        ('wiki_hop',WIKIHOP),
        ('yelp_review_full',YELPREVIEWFULL),
        ('dbpedia_14',DBPEDIA14),
        ('trec',TREC),
        ('imdb',IMDB),
        ('app_reviews',APPREVIEWS),
        ('gigaword',GIGAWORD),
        ('multi_news',MULTINEWS),
        ('rotten_tomatoes',ROTTENTOMATOES),
        ('wiki_auto',WIKIAUTO),
        ('asset',ASSET),
        ('ct0_gigaword',CT0_GIGAWORD),
        ('haiku',HAIKU),
        ('covid_qa',COVIDQA),
        ('eli5',ELI5),
        ('emdg',EMDG),
        ('esnli',eSNLI),
        ('twitter',TWITTER)
    ]
)

class GeneralTask(AbstractTask):
    def __init__(self, task, config, prompt, seed=42):
        self.task = task
        self.name = task
        self.config = config
        self.seed = seed
        self.prompt = prompt
        print('################')
        print(self.task)
        print()
        print(self.config)
        print()
        print(self.prompt)
        print('################')
        #if self.task[0:3] == "anli":
        #    self.promptList = DatasetTemplates("anli")[self.prompt]
        if self.config=="skip":
            print('Pass through ')
        else:
            if self.config!='none':
                self.promptList = DatasetTemplates(f"{task}/{config}")[self.prompt]
            else:
                self.promptList = DatasetTemplates(f"{task}")[self.prompt]
            
            name = task
            metric = []
            
            if task=="trivia_qa":
                metric_names = ["squad"]
                metric.append(metrics.squad)
            else:
                metric_names = [x.lower() for x in self.promptList.metadata.metrics]
                metric.append(metrics.accuracy)
                metric.append(metrics.calculate_rouge)
                metric.append(metrics.bleu)
                for x in metric_names:
                    if x =="other":
                        continue
                    elif x=="matthews_correlation":
                        metric.append(metrics.matthews_corrcoef)
                    elif x=="em":
                        metric.append(metrics.exact_match)
                    elif x=="f1":
                        metric.append(metrics.f1_score_with_invalid)
                    elif x=="spearmanr":
                        metric.append(metrics.spearman_corrcoef)
                    elif x=="f1_multiclass":
                        metric.append(metrics.mean_multiclass_f1(num_classes=3))
                    elif x=="squad":
                        metric.append(metrics.squad)
            self.metric_names = metric_names
            self.metric = metric
            split_to_data_split = {"train": "train",
                                "validation": "validation",
                                "test": "test"}

    def load_dataset(self, split: str):
        print('####')
        print(self.task)
        print('####')
        if self.task=="anli":
            return datasets.load_dataset(self.task,cache_dir='cache',script_version="master")[self.config]
        elif self.task=='rotten_tomatoes':
            x = datasets.load_dataset(self.task,split=split,cache_dir='cache',script_version='master')
            x = x.rename_column('label','labels')
            return x
        elif self.task=="app_reviews":
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version="master")
            x['validation'] = x['train'][10000:10200]
            return x[split]
        elif self.task=="wiki_bio":
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version="master")
            x['validation'] = x['val']
            return x[split]
        elif self.task =="yelp_review_full":
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version="master")
            x['validation'] = x['test']
            #x = x.rename_column('label','labels')
            x = x.remove_columns("label")
            return x[split]
        elif self.task == "imdb":
            print('####')
            print(self.task)
            print('####')
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version="master")
            x['validation'] = x['test']
            #x = x.rename_column('label','labels')
            #x['labels'] = x['label']
            return x[split]
        elif self.task == "ag_news":
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version='master')
            x['validation'] = x['test']
            return x[split]
        elif self.task == "dbpedia_14":
            x = datasets.load_dataset(self.task)
            x['validation'] = x['test']
            return x[split]
        elif self.task == "trec":
            x = datasets.load_dataset(self.task,cache_dir='cache',script_version='master')
            x['validation'] = x['test']
            return x[split]
        elif self.config!="none":
            x = datasets.load_dataset(self.task, self.config, split=split,cache_dir='cache',script_version="master")
            return x
        elif self.task=="super_glue":
            if self.config=="copa_gen":
                return datasets.load_dataset(self.task,"copa",cache_dir='cache',split=split, script_version="master")
        elif self.task=='xsum':
            return datasets.load_dataset("xsum",split=split)
        else:
            #return datasets.load_dataset(self.task, split=split,cache_dir='cache', script_version="master",download_mode="force_redownload")
            return datasets.load_dataset(self.task, split=split,cache_dir='cache', script_version="master")
            #return datasets.load_dataset(self.task, split=split)

    def preprocessor(self, example, add_prefix=True):
        tmp = self.promptList.apply(example)

        src_texts = tmp[:-1]
        tgt_texts = tmp[-1]

        #result = self.seq2seq_format(src_texts, tgt_texts, add_prefix=False)
        result = {}
        result['source'] = ' '.join(src_texts)
        result['target'] = tgt_texts
        result['task'] = self.name
        result['extra_fields'] = {}
        # if self.name == 'commonsense_qa':
        #     if self.prompt == 'question_answering':
        #         result['labels_list'] = [item for item in example['choices']['text']]
        # elif self.name == 'dream':
        #     if self.prompt == 'read_the_following_conversation_and_answer_the_question':
        #         result['labels_list'] = [item for item in example['choice']]
        # # all prompts are multi choice
        # elif self.name == 'quail':
        #     result['labels_list'] = [item for item in example['answers']]
        # # all prompts are multi choice
        # elif self.name == 'quartz':
        #     result['labels_list'] = [item for item in example['choices']['text']]
        # elif self.name == 'social_i_qa':
        #     if self.prompt == 'Show choices and generate answer':
        #         result['labels_list'] = [example['answerA'], example['answerB'], example['answerC']]
        if "super_glue" in self.name:

            if self.config == "copa":
                result['labels_list'] = [example['choice1'],example['choice2']]
            elif self.config == "cb":
                if self.prompt == "can we infer":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "claim true/false/inconclusive":
                    result['labels_list'] = ["True","Inconclusive","False"]
                elif self.prompt == "MNLI crowdsource":
                    result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
                elif self.prompt == "should assume":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "does it follow that":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "GPT-3 style":
                    result['labels_list'] = ["True","False","Neither"]
                elif self.prompt == "based on the previous passage":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "justified in saying":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "take the following as truth":
                    result['labels_list'] = ["True","Inconclusive","False"]
                elif self.prompt == "must be true":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "guaranteed/possible/impossible":
                    result['labels_list'] = ["Guaranteed","Possible","Impossible"]
                elif self.prompt == "always/sometimes/never":
                    result['labels_list'] = ["Always","Sometimes","Never"]
                elif self.prompt == "does this imply":
                    result['labels_list'] = ["Yes","Maybe","No"]
                elif self.prompt == "consider always/sometimes/never":
                    result['labels_list'] = ["Always","Sometimes","Never"]
                elif self.prompt == "guaranteed true":
                    result['labels_list'] = ["Yes","Maybe","No"]
            elif self.config == "rte":
                if self.prompt == "GPT-3 style":
                    result['labels_list'] = ["True","False"]
                else:
                    result['labels_list'] = ["Yes","No"]
            elif self.config == 'wsc.fixed':
                if self.prompt == "p is/are r":
                    result['labels_list'] = ["False",'True']
                elif self.prompt == "the pronoun refers to":
                    result['labels_list'] = ["False",'True']
                elif self.prompt == "in other words":
                    result['labels_list'] = ["False",'True']
                else:
                    result['labels_list'] = ["No", "Yes"]
            elif self.config == "wic":
                if self.prompt == "affirmation_true_or_false":
                    result['labels_list'] = ["False","True"]
                else:
                    result['labels_list'] = ["No","Yes"]
        elif self.name == "winogrande":
            if self.config =="winogrande_xl":
                if self.prompt == "True or False":
                    result['labels_list'] = ["True","False"]
                else:
                    result['labels_list'] = [example['option1'],example['option2']]
        elif self.name == "hellaswag":
            if self.prompt == "Predict ending with hint":
                result['labels_list'] = [item for item in example['endings']]
            elif self.prompt == "complete_first_then":
                result['labels_list'] = [item for item in example['endings']]
            elif self.prompt == "Randomized prompts template":
                result['labels_list'] = [item for item in example['endings']]
            elif self.prompt == "Appropriate continuation - Yes or No":
                result['labels_list'] = ["Yes","No"]
            elif self.prompt == "Reversed appropriate continuation - Yes or No":
                result['labels_list'] = ["Yes","No"]
            elif self.prompt == "how_ends":
                result['labels_list'] = ['Ending 1','Ending 2','Ending 3','Ending 4']
            elif self.prompt == "if_begins_how_continues":
                result['labels_list'] = ['Ending 1','Ending 2','Ending 3','Ending 4']
        #elif self.name =="story_cloze":
        #    result['labels_list'] = [example['RandomFifthSentenceQuiz1'],example['RandomFifthSentenceQuiz2']]
        elif self.name == 'anli_r1':
            if self.prompt == "can we infer":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "claim true/false/inconclusive":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "MNLI crowdsource":
                result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
            elif self.prompt == "should assume":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "does it follow that":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "GPT-3 style":
                result['labels_list'] = ["True","False","Neither"]
            elif self.prompt == "based on the previous passage":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "justified in saying":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "take the following as truth":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "must be true":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "guaranteed/possible/impossible":
                result['labels_list'] = ["Guaranteed","Possible","Impossible"]
            elif self.prompt == "always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "does this imply":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "consider always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "guaranteed true":
                result['labels_list'] = ["Yes","Maybe","No"]
        elif self.name == 'anli_r2':
            if self.prompt == "can we infer":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "claim true/false/inconclusive":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "MNLI crowdsource":
                result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
            elif self.prompt == "should assume":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "does it follow that":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "GPT-3 style":
                result['labels_list'] = ["True","False","Neither"]
            elif self.prompt == "based on the previous passage":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "justified in saying":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "take the following as truth":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "must be true":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "guaranteed/possible/impossible":
                result['labels_list'] = ["Guaranteed","Possible","Impossible"]
            elif self.prompt == "always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "does this imply":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "consider always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "guaranteed true":
                result['labels_list'] = ["Yes","Maybe","No"]
        elif self.name == 'anli_r3':
            if self.prompt == "can we infer":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "claim true/false/inconclusive":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "MNLI crowdsource":
                result['labels_list'] = ["Correct","Inconclusive","Incorrect"]
            elif self.prompt == "should assume":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "does it follow that":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "GPT-3 style":
                result['labels_list'] = ["True","False","Neither"]
            elif self.prompt == "based on the previous passage":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "justified in saying":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "take the following as truth":
                result['labels_list'] = ["True","Inconclusive","False"]
            elif self.prompt == "must be true":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "guaranteed/possible/impossible":
                result['labels_list'] = ["Guaranteed","Possible","Impossible"]
            elif self.prompt == "always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "does this imply":
                result['labels_list'] = ["Yes","Maybe","No"]
            elif self.prompt == "consider always/sometimes/never":
                result['labels_list'] = ["Always","Sometimes","Never"]
            elif self.prompt == "guaranteed true":
                result['labels_list'] = ["Yes","Maybe","No"]
        if use_verbalizer:
            if self.name == "cos_e":
                if self.prompt == "question_description_option_text":
                    result['labels_list'] = [item for item in example['choices']]
                elif self.prompt=="question_description_option_id":
                    result['labels_list'] = ["A","B","C","D","E"]
                elif self.prompt== "question_option_description_text":
                    result['labels_list'] = [item for item in example['choices']]
                elif self.prompt== "description_question_option_id":
                    result['labels_list'] = ["A","B","C","D","E"]
                elif self.prompt== "description_question_option_text":
                    result['labels_list'] = [item for item in example['choices']]
                elif self.prompt== "question_option_description_id":
                    result['labels_list'] = ["A","B","C","D","E"]
            elif self.name=="commonsense_qa":
                if self.prompt == "answer_given_question_without_options":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "question_answering":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "question_to_answer_index":
                    result['labels_list'] = [item for item in example['choices']['label']]
                elif self.prompt == "most_suitable_answer":
                    result['labels_list'] = [item for item in example['choices']['text']]
            elif self.name=="dream":
                if self.prompt == "baseline":
                    result['labels_list'] = [item for item in example['choice']]
                elif self.prompt == "read_the_following_conversation_and_answer_the_question":
                    result['labels_list'] = [item for item in example['choice']]
            elif self.name=="quail":
                if self.prompt == "context_question_answer_description_id":
                    result['labels_list'] = ["A","B","C","D"]
                elif self.prompt == "context_question_answer_description_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "description_context_question_answer_id":
                    result['labels_list'] = ["A","B","C","D"]
                elif self.prompt == "context_question_description_answer_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "context_question_description_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "context_description_question_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "context_question_description_answer_id":
                    result['labels_list'] = ["A","B","C","D"]
                elif self.prompt == "no_prompt_id":
                    result['labels_list'] = ["A","B","C","D"]
                elif self.prompt == "context_description_question_answer_id":
                    result['labels_list'] = ["A","B","C","D"]
                elif self.prompt == "description_context_question_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "no_prompt_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "context_description_question_answer_text":
                    result['labels_list'] = [item for item in example['answers']]
                elif self.prompt == "description_context_question_answer_text":
                    result['labels_list'] = [item for item in example['answers']]
            elif self.name=="quartz":
                if self.prompt == "use_info_from_question_paragraph":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "paragraph_question_plain_concat":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "use_info_from_paragraph_question":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "answer_question_based_on":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "answer_question_below":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "read_passage_below_choose":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "having_read_above_passage":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "given_the_fact_answer_the_q":
                    result['labels_list'] = [item for item in example['choices']['text']]
            elif self.name=="social_i_qa":
                if self.prompt == "I was wondering":
                    result['labels_list'] = [example['answerA'],example['answerB'],example['answerC']]
                elif self.prompt == "Show choices and generate answer":
                    result['labels_list'] = [example['answerA'],example['answerB'],example['answerC']]
                elif self.prompt == "Check if a random answer is valid or not":
                    result['labels_list'] = ['Yes','No']
                elif self.prompt == "Generate answer":
                    result['labels_list'] = [example['answerA'],example['answerB'],example['answerC']]
                elif self.prompt == "Show choices and generate index":
                    result['labels_list'] = ['A','B','C']
            elif self.name=="wiqa":
                if self.prompt == "effect_with_string_answer":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "which_of_the_following_is_the_supposed_perturbation":
                    result['labels_list'] = ["indirectly impacting a step of the process", "not impacting any step of the process"]
                elif self.prompt == "effect_with_label_answer":
                    result['labels_list'] = ["A","B","C"]
                elif self.prompt == "does_the_supposed_perturbation_have_an_effect":
                    result['labels_list'] = ["yes","no"]
            elif self.name=="cosmos_qa":
                if self.prompt == "description_context_question_answer_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "description_context_question_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "description_context_question_answer_id":
                    result['labels_list'] = ['A','B','C','D']
                elif self.prompt == "context_description_question_answer_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "no_prompt_id":
                    result['labels_list'] = ['A','B','C','D']
                elif self.prompt == "context_question_description_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "no_prompt_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "context_description_question_answer_id":
                    result['labels_list'] = ['A','B','C','D']
                elif self.prompt == "context_question_description_answer_id":
                    result['labels_list'] = ['A','B','C','D']
                elif self.prompt == "context_description_question_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "context_question_description_answer_text":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
                elif self.prompt == "only_question_answer":
                    result['labels_list'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
            elif self.name=="qasc":
                if self.prompt == "is_correct_1":
                    result['labels_list'] = ['Yes','No']
                elif self.prompt == "qa_with_separated_facts_1":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "qa_with_separated_facts_3":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "qa_with_separated_facts_4":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "qa_with_separated_facts_5":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "qa_with_combined_facts_1":
                    result['labels_list'] = [item for item in example['choices']['text']]
                elif self.prompt == "is_correct_2":
                    result['labels_list'] = ['Yes','No']
                elif self.prompt == "qa_with_separated_facts_2":
                    result['labels_list'] = [item for item in example['choices']['text']]
            elif self.name=="quarel":
                if self.prompt == "do_not_use":
                    result['labels_list'] = [example['world_literals']['world1'][0], example['world_literals']['world2'][0]]
                elif self.prompt == "logic_test":
                    result['labels_list'] = [example['world_literals']['world1'][0], example['world_literals']['world2'][0]]
                elif self.prompt == "heres_a_story":
                    result['labels_list'] = [example['world_literals']['world1'][0], example['world_literals']['world2'][0]]
                elif self.prompt == "choose_between":
                    result['labels_list'] = [example['world_literals']['world1'][0], example['world_literals']['world2'][0]]
                elif self.prompt == "testing_students":
                    result['labels_list'] = [example['world_literals']['world1'][0], example['world_literals']['world2'][0]]
            elif self.name=="sciq":
                if self.prompt == "Direct Question (Closed Book)":
                    result['labels_list'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
                elif self.prompt == "Multiple Choice (Closed Book)":
                    result['labels_list'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
                elif self.prompt == "Multiple Choice Question First":
                    result['labels_list'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
                elif self.prompt == "Multiple Choice":
                    result['labels_list'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
                elif self.prompt == "Direct Question":
                    result['labels_list'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
            
            elif self.name == "app_reviews":
                if self.prompt == "categorize_rating_using_review":
                    result['labels_list'] = ["Not at all","No","Maybe","Yes","Definitely"]
                elif self.prompt == "convert_to_star_rating":
                    result['labels_list'] = ["\u2605","\u2605\u2605","\u2605\u2605\u2605","\u2605\u2605\u2605\u2605","\u2605\u2605\u2605\u2605\u2605"]
            elif self.name == "imdb":
                if self.prompt == "Movie Expressed Sentiment 2":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Opinion bad good choices":
                    result['labels_list'] = ["bad","good"]
                elif self.prompt == "Sentiment with choices ":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Sentiment Feeling":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Writer Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Movie Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Text Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Negation template for positive and negative":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Enjoyment Yes No":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Reviewer Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Enjoyment":
                    result['labels_list'] = ["They didn't like it!","They loved it"]
            elif self.name == "rotten_tomatoes":
                if self.prompt == "Reviewer Opinion bad good choices":
                    result['labels_list'] = ["bad","good"]
                elif self.prompt == "Text Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Sentiment with choices ":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Enjoyment Yes No":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Reviewer Enjoyment":
                    result['labels_list'] = ["They didn't like it","They loved it"]
                elif self.prompt == "Movie Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Writer Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Movie Expressed Sentiment 2":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Expressed Sentiment":
                    result['labels_list'] = ["negative","positive"]
                elif self.prompt == "Reviewer Sentiment Feeling":
                    result['labels_list'] = ["negative","positive"]
            elif self.name == "paws":
                if self.prompt == "task_description-no-label":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Meaning":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "context-question-no-label":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Rewrite-no-label":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "context-question":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Concatenation":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Concatenation-no-label":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "Meaning-no-label":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "PAWS-ANLI GPT3":
                    result['labels_list'] = ["False","True"]
                elif self.prompt == "Rewrite":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "PAWS-ANLI GPT3-no-label":
                    result['labels_list'] = ["No","Yes"]
            elif self.name == "glue_qqp":
                if self.prompt == "quora":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "duplicate or not":
                    result['labels_list'] = ["not duplicates","duplicates"]
                elif self.prompt == "same thing":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "answer":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "meaning":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt == "duplicate":
                    result['labels_list'] = ["no","yes"]
            elif self.name == "glue_mrpc":
                if self.prompt == "want to know":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "paraphrase":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "equivalent":
                    result['labels_list'] = ["not equivalent","equivalent"]
                elif self.prompt == "replace":
                    result['labels_list'] = ["no","yes"]
                elif self.prompt == "same thing":
                    result['labels_list'] = ["no","yes"]
            elif self.name == "ag_news":
                if self.prompt == "classify_question_first":
                    result['labels_list'] = ["World politics","Sports","Business","Science and technology"]
                elif self.prompt == "classify_with_choices_question_first":
                    result['labels_list'] = ["World politics","Sports","Business","Science and technology"]
                elif self.prompt == "recommend":
                    result['labels_list'] = ["Politician","Athlete","Business executive","Scientist"]
                elif self.prompt == "which_section_choices":
                    result['labels_list'] = ["World News","Sports","Business","Science and Technology"]
                elif self.prompt == "which_section":
                    result['labels_list'] = ["World News","Sports","Business","Science and Technology"]
                elif self.prompt == "classify_with_choices":
                    result['labels_list'] = ["World politics","Sports","Business","Science and technology"]
                elif self.prompt == "classify":
                    result['labels_list'] = ["World politics","Sports","Business","Science and technology"]
            elif self.name == "dbpedia_14":
                if self.prompt == "given_list_what_category_does_the_paragraph_belong_to":
                    result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
                elif self.prompt == "pick_one_category_for_the_following_text":
                    result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
                elif self.prompt == "given_a_choice_of_categories":
                    result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
                elif self.prompt == "given_a_list_of_category_what_does_the_title_belong_to":
                    result['labels_list'] = ["Company","Educational Institution","Artist","Athlete","Office Holder","Mean Of Transportation","Building","Natural Place","Village","Animal","Plant","Album","Film","Written Work"]
            elif self.name == "trec":
                if self.prompt == "what_category_best_describe":
                    result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
                elif self.prompt == "fine_grained_LOC":
                    result['labels_list'] = ["city","country","mountain","state","other location"]
                elif self.prompt == "fine_grained_NUM_context_first":
                    result['labels_list'] = ["code","count","date","distance","price","order","period of time","percentage","speed","temperature","size","weight","other number"]
                elif self.prompt == "fine_grained_ENTY":
                    result['labels_list'] = ["an animal","an organ of the body","a color","creative piece","currency","disease or medicine","event","food","musical instrument","language","letter","plant","product","religion","sport","substance","symbol","technique","term","vehicle","word","other entity"]
                elif self.prompt == "fine_grained_NUM":
                    result['labels_list'] = ["code","count","date","distance","price","order","period of time","percentage","speed","temperature","size","weight","other number"]
                elif self.prompt == "pick_the_best_descriptor":
                    result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
                elif self.prompt == "fine_grained_open_context_first":
                    result['labels_list'] = ["Manner","Creative Piece","Animal","Expression abbreviated","Individual","Group","Title","Defintion","Date","Reason","Event","State","Description","Count","Other","Letter","Religion","Food","Country","Color","Term","City","Organ of the body","Disease or medicine","Mountain","Price","Product","Period","Substance","Sport","Plant","Technique","Size","Instrument","Abbreviation","Speed","Word","Language","Percentage","Code","Distance","Temperature","Symbol","Order","Vehicle","Weight","Currency"]
                elif self.prompt == "fine_grained_LOC_context_first":
                    result['labels_list'] = ["city","country","mountain","state","other location"]
                elif self.prompt == "which_category_best_describes":
                    result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
                elif self.prompt == "fine_grained_DESC":
                    result['labels_list'] = ["definition","description","manner of action","reason"]
                elif self.prompt == "trec1":
                    result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
                elif self.prompt == "fine_grained_ABBR":
                    result['labels_list'] = ["abbreviation","expression abbreviated"]
                elif self.prompt == "fine_grained_ABBR_context_first":
                    result['labels_list'] = ["abbreviation","expression abbreviated"]
                elif self.prompt == "trec2":
                    result['labels_list'] = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
                elif self.prompt == "fine_grained_HUM":
                    result['labels_list'] = ["group","individual","title","description"]
                elif self.prompt == "fine_grained_open":
                    result['labels_list'] = ["Manner","Creative Piece","Animal","Expression abbreviated","Individual","Group","Title","Defintion","Date","Reason","Event","State","Description","Count","Other","Letter","Religion","Food","Country","Color","Term","City","Organ of the body","Disease or medicine","Mountain","Price","Product","Period","Substance","Sport","Plant","Technique","Size","Instrument","Abbreviation","Speed","Word","Language","Percentage","Code","Distance","Temperature","Symbol","Order","Vehicle","Weight","Currency"]
                elif self.prompt == "fine_grained_HUM_context_first":
                    result['labels_list'] = ["group","individual","title","description"]
                elif self.prompt == "fine_grained_DESC_context_first":
                    result['labels_list'] = ["definition","description","manner of action","reason"]
            elif self.name=="hotpot_qa":
                if self.prompt =="classify_question_type":
                    result['labels_list'] = ["comparison","bridge"]
            elif self.name=="wiki_qa":
                if self.prompt=="Is This True?":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt=="automatic_system":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt=="found_on_google":
                    result['labels_list'] = ["No","Yes"]
                elif self.prompt=="exercise":
                    result['labels_list'] = ["False","True"]
                elif self.prompt=="Decide_good_answer":
                    result['labels_list'] = ["No","Yes"]
        return result


class AutoTask:
    @classmethod
    def get(self, task, config, prompt, seed=42):
        if task=="lama":
            return TASK_MAPPING["lama"](config, seed)
        elif task=="story_cloze":
            return TASK_MAPPING["story_cloze"](config, prompt, seed)
        elif task=="anli_r1":
            return TASK_MAPPING["anli_dev_r1"](config, prompt, seed)
        elif task=="anli_r2":
            return TASK_MAPPING["anli_dev_r2"](config, prompt, seed)
        elif task=="anli_r3":
            return TASK_MAPPING["anli_dev_r3"](config, prompt, seed)
        elif task=="super_glue_rte":
            return GeneralTask('super_glue','rte',prompt,seed)
        elif task=="super_glue_copa":
            return GeneralTask('super_glue','copa',prompt,seed)
        elif task=="super_glue_cb":
            return GeneralTask('super_glue','cb',prompt,seed)
        elif task=="super_glue_wsc.fixed":
            return GeneralTask('super_glue','wsc.fixed',prompt,seed)
        elif task=="super_glue_wic":
            return GeneralTask('super_glue','wic',prompt,seed)
        elif task=="glue_qqp":
            return GeneralTask('glue','qqp',prompt,seed)
        elif task=='glue_mrpc':
            return GeneralTask('glue','mrpc',prompt,seed)
        elif task=='wiki_hop':
            return TASK_MAPPING['wiki_hop'](config, prompt, seed)
        elif task=='amazon_polarity':
            return TASK_MAPPING['amazon_polarity'](config, prompt, seed)
        elif task=='yelp_review_full':
            return TASK_MAPPING['yelp_review_full'](config, prompt, seed)
        elif task=='dbpedia_14':
            return TASK_MAPPING['dbpedia_14'](config, prompt, seed)
        elif task=='trec':
            return TASK_MAPPING['trec'](config, prompt, seed)
        elif task=='imdb':
            return TASK_MAPPING['imdb'](config, prompt, seed)
        elif task=='app_reviews':
            return TASK_MAPPING['app_reviews'](config, prompt, seed)
        elif task=='gigaword':
            return TASK_MAPPING['gigaword'](config, prompt, seed)
        elif task=='multi_news':
            return TASK_MAPPING['multi_news'](config, prompt, seed)
        elif task=='rotten_tomatoes':
            return TASK_MAPPING['rotten_tomatoes'](config, prompt, seed)
        elif task=='wiki_auto':
            return TASK_MAPPING['wiki_auto'](config, prompt, seed)
        elif task=="asset":
            return TASK_MAPPING['asset'](config, prompt, seed)
        elif task=="ct0_gigaword":
            return TASK_MAPPING['ct0_gigaword'](config, prompt, seed)
        elif task=="haiku":
            return TASK_MAPPING['haiku'](config, prompt, seed)
        elif task=="covid_qa":
            return TASK_MAPPING['covid_qa'](config, prompt, seed)
        elif task=="eli5":
            return TASK_MAPPING['eli5'](config, prompt, seed)
        elif task=="emdg":
            return TASK_MAPPING['emdg'](config, prompt, seed)
        elif task=="esnli":
            return TASK_MAPPING['esnli'](config, prompt, seed)
        elif task=="twitter":
            return TASK_MAPPING['twitter'](config, prompt, seed)
        else:
            print('$$$$$$$$$$$$$$$$')
            print(task)
            print()
            print(config)
            print()
            print(prompt)
            print('$$$$$$$$$$$$$$$$')
            return GeneralTask(task, config, prompt, seed)
        raise ValueError(
            f"Task {task}/{config}/{prompt} was not successfully loaded."
        )