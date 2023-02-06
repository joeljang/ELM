import os, json
import numpy as np
from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
import _pickle as cPickle

class Clf():
  
  """
  Usage:
  1) load the clf for a task:
  path_folder_data = f'{GLOBAL_PATH}/data'
  evalset = 'twitter_top20'
  prompt_name = 'tweet_as+about'
  label_name = 'author'
  clf = Clf(path_folder_data, evalset, prompt_name, label_name)
  
  2) infer:
  print(clf.compute_score(evaluated_predictions))
  """

  def __init__(self, path_folder_data, evalset, prompt_name, label_name):
    self.path_folder_data = path_folder_data
    self.evalset = evalset
    self.prompt_name = prompt_name
    self.label_name = label_name

    self.key_name = f'{evalset}.{prompt_name}.{label_name}'

    path_model = f'{self.key_name}.model.pkl'
    path_count_vectorizer = f'{self.key_name}.count_vectorizer.pkl'

    if os.path.exists(path_model):
      # load it
      with open(path_model, 'rb') as fid:
          self.model = cPickle.load(fid)
      with open(path_count_vectorizer, 'rb') as fid:
          self.count_vectorizer = cPickle.load(fid)
    else:
      self.model = RidgeClassifier() #GaussianNB()
      self.count_vectorizer = CountVectorizer(binary=True)
      self.train_model()
      # save the classifier
      with open(path_model, 'wb') as fid:
        cPickle.dump(self.model, fid)  
      with open(path_count_vectorizer, 'wb') as fid:
        cPickle.dump(self.count_vectorizer, fid)  

    #transform test data
    X_test, y_test = self.get_data('test')
    self.y_test = y_test
    predictions = self.get_preds(X_test)
    print("Accuracy clf:", self.accuracy_score(y_test, predictions))    

  def get_data(self, eval_mode):

    path_ex = os.path.join(self.path_folder_data, self.evalset, f'{self.prompt_name}.{eval_mode}.json')

    with open(path_ex, 'r') as f:
      data = json.load(f)

    nb_ex = len(data['src_info'])
    outputs = [data['tgt'][idx] for idx in range(nb_ex)]
    labels = [data['src_info'][idx][self.label_name] for idx in range(nb_ex)]
    
    assert len(outputs) == len(labels)

    return outputs, labels

  def train_model(self):
    
    #fit training data
    X_train, y_train = self.get_data('train')
    training_data = self.count_vectorizer.fit_transform(X_train).toarray()
    self.model.fit(training_data, y_train)

  @staticmethod
  def accuracy_score(y_true, y_pred):
    return np.average([y1 == y2 for y1, y2 in zip(y_true, y_pred)])

  def get_preds(self, X_test):
    testing_data = self.count_vectorizer.transform(X_test).toarray()
    predictions = self.model.predict(testing_data)

    return predictions

  def compute_score(self, outputs):

    clf_predictions = self.get_preds(outputs)
    print('*****************************')
    print(len(clf_predictions))
    print(len(self.y_test))
    print('*****************************')
    return {'CLF_acc': self.accuracy_score(self.y_test, clf_predictions)}

path_folder_data = "/home/joel_jang/seungone/RoE/seq2seq/data/manual/ct0_data/twitter_top20"
evalset = "twitter_top20"
prompt_name = "tweet_as+about"
label_name = "author"

clf = Clf(path_folder_data,evalset,prompt_name,label_name)

twitter_top20 = "/home/joel_jang/seungone/RoE/seq2seq/output_logs/twitter/twitter*tweet_as+about-twitter*tweet_as+about.txt"
results = [twitter_top20]

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

results = clf.compute_score(predictions[0])
print(results)