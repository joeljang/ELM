from adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    wandb_log: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, logs experimental results to wandb"})
    wandb_entity: Optional[str] = field(default="lklab_kaist",
                                                metadata={"help": "Set to the the wandb name"})
    wandb_project: Optional[str] = field(default="retrieval_of_experts",
                                                metadata={"help": "Set to the project name of wandb"})
    wandb_run_name: Optional[str] = field(default="default_run_yolo",
                                                metadata={"help": "Desingate the wandb run name"})
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of the model."})
    do_train: Optional[bool] = field(default=False, metadata={
                                    "help": "If set, evaluates the train performance."})
    do_eval: Optional[bool] = field(default=False, metadata={
                                    "help": "If set, evaluates the eval performance."})
    do_test: Optional[bool] = field(default=False, metadata={
                                    "help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(
        default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(
        default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(
        default=100, metadata={"help": "Defines the length for prefix tuning."})
    report_to: Optional[str] = field(default="wandb")
    save_strategy: Optional[str] = field(default="no")

