import evaluate
from data.tasks import AutoTask, TASK_MAPPING, STORYCLOZE, ANLIR1, ANLIR2, ANLIR3, WIKIHOP, AMAZONPOLARITY, YELPREVIEWFULL, DBPEDIA14, TREC, IMDB, APPREVIEWS, GIGAWORD, ROTTENTOMATOES, GeneralTask
import torch
import json
from tqdm import tqdm
import syllables
import numpy as np
import scipy

#wiki auto, asset
print("WIKI AUTO / ASSET")
# wiki_auto_simp1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/wiki_auto/wiki_auto*simplification_1-wiki_auto*simplification_1.txt"
# wiki_auto_simp2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/wiki_auto/wiki_auto*simplification_1-wiki_auto*simplification_2.txt"
# asset_simp1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/wiki_auto/wiki_auto*simplification_1-asset*simplification_1.txt"
# asset_simp2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/wiki_auto/wiki_auto*simplification_1-asset*simplification_2.txt"
wiki_auto_simp1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-wiki_auto*simplification_1.txt"
wiki_auto_simp2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-wiki_auto*simplification_2.txt"
asset_simp1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-asset*simplification_1.txt"
asset_simp2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-asset*simplification_2.txt"
results = [wiki_auto_simp1,wiki_auto_simp2,asset_simp1,asset_simp2]

predictions = []
references = []
sources = []

for idx,r in enumerate(results):
    pred = []
    ref = []
    src = []
    with open(r,'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('##' not in line) and ('>>' not in line) and ('*' not in line):

                p = line.split(' | ')[0].strip()
                r = line.split(' | ')[1].strip()
                s = line.split(' | ')[2].strip()
                pred.append(p)
                ref.append([r])
                src.append(s)
    predictions.append(pred)
    references.append(ref)
    sources.append(src)

bleu = evaluate.load("bleu")
sari = evaluate.load("sari")
for p,r,s in zip(predictions,references,sources):
    results = bleu.compute(predictions=p, references=r, max_order=4)
    print(results)    

    results = sari.compute(sources=s, predictions=p, references=r)
    print(results)

# gigaword
print()
print("GIGAWORD")
# gigaword_contain_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_contain+make_a_title.txt"
# gigaword_contain_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_contain+write_its_sentence.txt"

# gigaword_start_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_start+make_a_title.txt"
# gigaword_start_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_start+write_its_sentence.txt"

# gigaword_end_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_end+make_a_title.txt"
# gigaword_end_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/ct0_gigaword/ct0_gigaword*constrain_contain+make_a_title-ct0_gigaword*constrain_end+write_its_sentence.txt"
gigaword_contain_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_contain+make_a_title.txt"
gigaword_contain_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_contain+write_its_sentence.txt"

gigaword_start_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_start+make_a_title.txt"
gigaword_start_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_start+write_its_sentence.txt"

gigaword_end_prompt1 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_end+make_a_title.txt"
gigaword_end_prompt2 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-ct0_gigaword*constrain_end+write_its_sentence.txt"
results = [gigaword_contain_prompt1,gigaword_contain_prompt2,gigaword_start_prompt1,gigaword_start_prompt2,gigaword_end_prompt1,gigaword_end_prompt2]

predictions = []
references = []
sources = []

for idx,r in enumerate(results):
    pred = []
    ref = []
    src = []
    with open(r,'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('##' not in line) and ('>>' not in line) and ('*' not in line):

                p = line.split(' | ')[0].strip()
                r = line.split(' | ')[1].strip()
                s = line.split(' | ')[2].strip()
                pred.append(p)
                ref.append([r])
                src.append(s)
    predictions.append(pred)
    references.append(ref)
    sources.append(src)

rouge = evaluate.load("rouge")
for idx,(p,r,s) in enumerate(zip(predictions,references,sources)):
    results = rouge.compute(predictions=p, references=r, use_stemmer=True)
    score=0
    for i,(check_sent,sourc) in enumerate(zip(p,s)):
        if idx==0 or idx==1:
            constrain_word = sourc.split('"')[1].replace('"','')
            if constrain_word in check_sent:
                score+=1
        if idx==2 or idx==3:
            constrain_word = sourc.split('"')[1].replace('"','')
            if constrain_word == check_sent[:len(constrain_word)]:
                score+=1
        if idx==4 or idx==5:
            constrain_word = sourc.split('"')[1].replace('"','')
            if constrain_word == check_sent[-len(constrain_word):]:
                score+=1
    print(score/300)
    print(results)

# haiku
print()
print("HAIKU")
#haiku_do_nothing = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/haiku/haiku*do_nothing-haiku*do_nothing.txt"
#haiku_do_nothing = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/twitter/twitter*tweet_as+about-haiku*do_nothing.txt"
haiku_do_nothing = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-haiku*do_nothing.txt"
results = [haiku_do_nothing]

predictions = []
references = []
sources = []

for idx,r in enumerate(results):
    pred = []
    ref = []
    src = []
    with open(r,'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('##' not in line) and ('>>' not in line) and ('*' not in line):

                p = line.split(' | ')[0].strip()
                r = line.split(' | ')[1].strip()
                s = line.split(' | ')[2].strip()
                pred.append(p)
                ref.append([r])
                src.append(s)
    predictions.append(pred)
    references.append(ref)
    sources.append(src)

bleu = evaluate.load("bleu")
for idx,(p,r,s) in enumerate(zip(predictions,references,sources)):
    results = bleu.compute(predictions=p, references=r, max_order=4)['bleu']
    normaliseDifScore = lambda nb_tgt, nb_hyp: 1-abs(nb_tgt - nb_hyp)/max([nb_tgt, nb_hyp])
    constrainScorer = lambda src, hyp: 1 if ' '.join(src.split("'")[1:]).strip() in hyp else 0

    d_score = {
        'syllable': 0,
        'comma': 0,
        'constrain': 0,
        'bleu': results
    }

    for tgt, hyp, src in zip(r,p,s):
        tgt = tgt[0]
        d_score['syllable'] += normaliseDifScore(syllables.estimate(tgt), syllables.estimate(hyp)) 
        d_score['comma'] += normaliseDifScore(len(tgt.split(',')), len(hyp.split(','))) 
        d_score['constrain'] += constrainScorer(src, hyp) 

    for k in ['syllable', 'comma', 'constrain']:
        d_score[k] /= len(p)
    d_score['eq_weighted'] = sum(d_score.values()) / len(d_score)
    print(d_score)

# eli5
print()
print("ELI5")
#eli5_qgen = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_ct0/eli5/eli5*generate_a_question_1-eli5*generate_a_question_1.txt"
eli5_qgen = "/home/joel_jang/seungone/RoE/seq2seq/output_logs_t0_target/t0/t0*t0-eli5*generate_a_question_1.txt"
results = [eli5_qgen]

predictions = []
references = []
sources = []

for idx,r in enumerate(results):
    pred = []
    ref = []
    src = []
    with open(r,'r') as f:
        lines = f.readlines()
        for line in lines:
            if ('##' not in line) and ('>>' not in line) and ('*' not in line):

                p = line.split(' | ')[0].strip()
                r = line.split(' | ')[1].strip()
                s = line.split(' | ')[2].strip()
                pred.append(p)
                ref.append(r)
                src.append(s)
    predictions.append(pred)
    references.append(ref)
    sources.append(src)

class FirstWordSim():    
  
  def __init__(self):
    pass
  
  def compute(self, preds, refs):
    tok2idx = self.getTok2idx(preds + refs)
    d = self.jensen_shannon_distance(self.getArray(tok2idx, preds), self.getArray(tok2idx, refs))
    return {'jensenFirstToken': 1/d}
  
  def jensen_shannon_distance(self, p, q):
      """
      Thanks to @sourcedexter (https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d)
      method to compute the Jenson-Shannon Distance 
      between two probability distributions
      """

      # convert the vectors into numpy arrays in case that they aren't
      p = np.array(p)
      q = np.array(q)

      # calculate m
      m = (p + q) / 2

      # compute Jensen Shannon Divergence
      divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

      # compute the Jensen Shannon Distance
      distance = np.sqrt(divergence)

      return distance


  def getFirstTok(self, sent):
    tok = ""
    if sent:
      tok = sent.split()[0].lower()

    return tok

  def getTok2idx(self, all_sents):
    
    tok2idx = {}
    count = 0
    for sent in all_sents:

      tok = self.getFirstTok(sent)
      if tok not in tok2idx:
        tok2idx[tok] = count
        count += 1

    return tok2idx

  def getArray(self, tok2idx, sents):

    arr = [0] * len(tok2idx)

    for sent in sents:
      tok = self.getFirstTok(sent)
      arr[tok2idx[tok]] += 1

    return arr

fws = FirstWordSim()
results = fws.compute(predictions[0], references[0])
print(results)