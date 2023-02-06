import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq


# @dataclass
# class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
#    def check_uniqueness(self, samples):
#       assert len(np.unique(samples)) == 1
   
#    def __call__(self, features):
#       tasks = []
#       for d in features:
#          if type(d) is dict:
#             tasks.append(d.pop('task'))
#          else:
#             tasks.append(d['task'])
#             d.remove_columns('task')

#       self.check_uniqueness(tasks)
#       output = super().__call__(features)
#       output["task"] = tasks[0]
#       return output
@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   def check_uniqueness(self, samples):
      assert len(np.unique(samples)) == 1
   
   def __call__(self, features):
      print('#$$$$$$$$$$$### COLLATOR DEBUG #$############$$$')
      print(features[0].keys())
      print('#############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
      tasks = [d.pop('task') for d in features]
      labels_list_exist=False
      if 'labels_list' in features[0]:
         labels_list_exist=True
         labels_list = [d.pop('labels_list') for d in features]
      self.check_uniqueness(tasks)
      output = super().__call__(features)
      output["task"] = tasks[0]
      if labels_list_exist:
         output["labels_list"] = labels_list
      # print('#$$$$$$$$$$$### COLLATOR DEBUG2 #$############$$$')
      # print(output)
      # print('#############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
      return output