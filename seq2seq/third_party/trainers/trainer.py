from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np 
import time
import torch
import torch.nn as nn
import collections
from packaging import version
from torch.utils.data.dataset import Dataset
import math
import wandb
import functools
import os
import inspect
import datasets
from data import AutoTask
from transformers.utils import logging
import transformers
from transformers import Trainer
from transformers import logging
from transformers.trainer_utils import (
    speed_metrics,
    EvalLoopOutput,
    denumpify_detensorize
)
#TODO : Change manual setting
instance_eval_mode = False
# import fairscale
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
# from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
# from fairscale.nn.wrap import auto_wrap
# from fairscale.optim import OSS
# from fairscale.optim.grad_scaler import ShardedGradScaler

#from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    #get_module_class_from_name,
    find_batch_size,
    nested_numpify,
    nested_truncate,
    nested_concat,
    IterableDatasetShard,
    #smp_forward_backward, 
    #smp_forward_only, 
    #smp_gather, 
    #smp_nested_concat
)
from .trainer_utils import EvalPrediction
#FSDPOption
from transformers.debug_utils import (
    DebugOption,
    DebugUnderflowOverflow
)

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments

from transformers.utils.modeling_auto_mapping import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.integrations import deepspeed_init


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)

class BaseTrainer(Trainer):
    def __init__(self, evaluation_metrics={}, data_info=None, eval_datasets=None, data_args=None, *args, **kwargs):
        """When doing evaluation, it computes average of list of metrics 
        given in evaluation_metrics and adds it to the dictionary of results.
        Trainer class then use this average metric to save the best model."""
        super().__init__(*args, **kwargs)
        self.evaluation_metrics = evaluation_metrics 
        self.data_info = data_info
        self.eval_datasets = eval_datasets
        self.data_args = data_args
        self.eval_data_configs = data_args.eval_dataset_config_name
        
        self.txt_save_dir = data_args.txt_save_dir
        #self.eval_metrics_dict = {dataset_name: AutoTask.get(dataset_name, dataset_config_name, prompt=eval_prompt).metric
        #            for dataset_name, dataset_config_name, eval_prompt in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.eval_prompts)}
        self.eval_metrics_dict = evaluation_metrics

    def update_evaluation_metrics(self, evaluation_metrics):
        self.evaluation_metrics = evaluation_metrics
        
    def get_data_info(self, metric_key_prefix):
        """Returns the data information required to make the predictions/labels
        suitable for the evaluation."""
        if self.data_info is not None:
            return self.data_info[metric_key_prefix]
        return None     
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        task: str = None
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        print('************* START OF EVAL ****************')
        print(task)
        print()
        print(eval_dataset)
        print('********************************************')
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        # TODO :sanity check
        # if len(eval_dataset)>100:
        #     g = torch.Generator()
        #     g.manual_seed(self.data_args.data_seed)
        #     rp = torch.randperm(len(eval_dataset),generator=g).tolist()
        #     rp = rp[:100]
        #     eval_dataset = eval_dataset.select(rp)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        print('********************************$$$$$*******')
        print(eval_dataset[0])
        print(transformers.__file__)
        print(eval_dataloader.dataset[0])
        print('********************************$$$$$*******')

        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            task = task
        )
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, output.num_samples))

        # if len(self.evaluation_metrics) != 0:
        #     selected_metrics = [output.metrics[metric_key_prefix+"_"+k] for k in self.evaluation_metrics if metric_key_prefix+"_"+k in output.metrics]
        #     print('$%$%$%$%$%$%$%$%')
        #     print(selected_metrics)
        #     print('$%$%$%$%$%$%$%$%')
        #     assert len(selected_metrics) >= 1, "at least one metric should be selected to compute the average_metrics."
        #     output.metrics.update({metric_key_prefix+'_average_metrics': np.mean(selected_metrics)})         
        
        self.log(output.metrics)
        
        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids",'task'] + self.label_names))
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if 'labels_list' in dataset.column_names:
            ignored_columns.remove("labels_list")
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if 'labels_list' in dataset.column_names:
            columns += ['labels_list']

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        task: str = None
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        model_preds_host = None
        preds_host = None
        labels_host = None
        #input_ids_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_model_preds = None
        all_labels = None
        all_input_ids = dataloader.dataset['input_ids']
        all_input_ids = np.array(all_input_ids, dtype=object)
        print(all_input_ids.shape)
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop

        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
            loss = None
            model_predictions = None
            labels = None
            logits = None
            #input_ids = inputs['input_ids']
            # print('@@@@@@@@@@@@@@@@@@#######@@@@@@@@@@@@@@@@@@')
            # print(input_ids)
            # print()

            # Prediction step
            # print('$$$$ DEBUG 0 $$$')
            # print(inputs)
            # print('$$$$$$$$$$$$$$$$')
            if 'labels_list' in inputs:
                loss, model_predictions, labels = self.verbalizer_prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            else:
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # print('$$$$ DEBUG 1 $$$')
            # print(logits[0,:])
            # print()
            # print(labels[0,:])
            # print()
            # print(input_ids[0,:])
            # print('$$$$$$$$$$$$$$$$')
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if model_predictions is not None:
                model_predictions = self._pad_across_processes(model_predictions)
                model_predictions = self._nested_gather(model_predictions)
                model_preds_host = model_predictions if model_preds_host is None else nested_concat(model_preds_host, model_predictions, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # print('$$$$ DEBUG 2 $$$')
            # print(labels[0,:])
            # print(labels.size())
            # print(labels_host[0,:])
            # print(labels_host.size())
            # print()
            # print(logits[0,:])
            # print(logits.size())
            # print(preds_host[0,:])
            # print(preds_host.size())
            # print()

            # if input_ids is not None:
            #     print(input_ids[0,:])
            #     input_ids = self._pad_across_processes(input_ids)
            #     print(input_ids[0,:])
            #     input_ids = self._nested_gather(input_ids)
            #     print(input_ids[0,:])
            #     input_ids_host = input_ids if input_ids is None else nested_concat(input_ids_host, input_ids, padding_index=-100)
            #     print(input_ids_host.size())
            
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
            
            # print(input_ids.size())
            # print(input_ids_host.size())
            # print('$$$$$$$$$$$$$$$$')
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if model_preds_host is not None:
                    model_predictions = nested_numpify(model_preds_host)
                    all_model_preds = model_predictions if all_model_preds is None else nested_concat(all_model_preds, model_predictions, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # print('$$$$ DEBUG 3 $$$')
                # print(logits[0,:])
                # print(logits.size())
                # print(all_preds[0,:])
                # print(all_preds.size())
                # print()
                # print(labels[0,:])
                # print(labels.size())
                # print(all_labels[0,:])
                # print(all_labels.size())
                # print()
                
                
                
                # all_input_ids = (
                #     input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
                # )
                # print(all_input_ids[0,:])
                # print(all_input_ids.size())
                # print('$$$$$$$$$$$$$$$$')
                # Set back to None to begin a new accumulation
                losses_host, preds_host, model_preds_host, labels_host, input_ids_host = None, None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if model_preds_host is not None:
            model_predictions = nested_numpify(model_preds_host)
            all_model_preds = model_predictions if all_model_preds is None else nested_concat(all_model_preds, model_predictions, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # print('$$$$ DEBUG 4 $$$')
        # print(labels.shape)
        # print(all_labels.shape)
        # print()
        # print(logits.shape)
        # print(all_preds.shape)
        
        #all_input_ids = nested_numpify(all_input_ids)
        
        #all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
        
        # print(all_input_ids.shape)
        # print('$$$$$$$$$$$$$$$$')
        # all_input_ids = input_ids
        # print(all_input_ids)
        # print()
        # print(all_preds)
        # print()
        # print(all_labels)
        # print('@@@@@@@@@@@@@@@@@@#######@@@@@@@@@@@@@@@@@@')
        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples
        
        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_model_preds is not None:
            all_model_preds = nested_truncate(all_model_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # print('$$$$ DEBUG 5 $$$')
        # print(all_labels.shape)
        # print()
        # print(all_preds.shape)
        # print()
        # print(all_input_ids.shape)
        if all_input_ids is not None:
            all_input_ids = nested_truncate(all_input_ids, num_samples)

        # print(all_input_ids.shape)
        # print('$$$$$$$$$$$$$$$$')
        # Metrics!
        if 'labels_list' in inputs:
            
            #model_predictions = model_predictions.cpu().detach().numpy()
            #all_labels = np.array(all_labels)
            all_input_ids = np.where(all_input_ids != -100, all_input_ids,
                            self.tokenizer.pad_token_id)
            all_labels = np.where(all_labels != -100, all_labels,
                            self.tokenizer.pad_token_id)
            print('$$$$ DEBUG 6 $$$')
            print(all_labels.shape)
            print()
            print(all_model_preds.shape)
            all_model_preds = np.where(all_model_preds != -100, all_model_preds,
                            self.tokenizer.pad_token_id)
            print(all_model_preds.shape)
            print('$$$$$$$$$$$$$$$$')
            decoded_input_ids = self.tokenizer.batch_decode(all_input_ids,skip_special_tokens=True)
            decoded_preds = self.tokenizer.batch_decode(all_model_preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(all_labels, skip_special_tokens=True)
            decoded_input_ids = [ii.strip() for ii in decoded_input_ids]
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            metrics = {}
            task_name = task
            eval_metrics = self.eval_metrics_dict[task_name]
        
            if instance_eval_mode:
                if os.path.isdir(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}") == False:
                    os.mkdir(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}")
                ##################################*
                tmp_result = {}
                with open(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{self.data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{task_name.replace('/',' ').replace('-',' ')}.txt",'a') as f:
                    for idx, (a,b) in enumerate(zip(decoded_preds, decoded_labels)):
                        for metric in eval_metrics:
                            tmp_result.update(metric(a,b))
                    
                        #with open(f"./output_logs_task_full_lm2/{data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{task_name.replace('/',' ').replace('-',' ')}.txt",'a') as f:
                        f.write(str(idx))
                        f.write(" | ")
                        f.write(a)
                        f.write(" | ")
                        f.write(b)
                        f.write(" | ")
                        f.write(">> ")
                        for key,value in tmp_result.items():
                            f.write(str(key)+" : "+str(value)+" # ")
                        f.write("\n")
                        metrics.update(tmp_result)
                        tmp_result={}
                ##################################*
            else:
                for metric in eval_metrics:
                    metrics.update(metric(decoded_preds, decoded_labels))
                
                #with open(f"./output_logs_t0_unseen/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{self.data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{str(task_name).replace('/',' ').replace('-',' ')}.txt",'a') as f:
                #with open(f"./output_logs_t0_verbalizer/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{self.data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{str(task_name).replace('/',' ').replace('-',' ')}.txt",'a') as f:
                
                # if os.path.isdir(f"./output_logs/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}") == False:
                #     os.mkdir(f"./output_logs/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}")
                # with open(f"./output_logs/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{self.data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{str(task_name).replace('/',' ').replace('-',' ')}.txt",'a') as f:
                if os.path.isdir(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}") == False:
                    os.mkdir(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}")
                with open(f"./{self.txt_save_dir}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}/{self.data_args.dataset_name[0].replace('/',' ').replace('-',' ')}*{self.data_args.train_prompts[0].replace('/',' ').replace('-',' ')}-{str(task_name).replace('/',' ').replace('-',' ')}.txt",'a') as f:
                    f.write("####################\n")
                    f.write(task_name)
                    f.write("\n")
                    for a,b,c in zip(decoded_preds, decoded_labels, decoded_input_ids):
                        f.write(a)
                        f.write(" | ")
                        f.write(b)
                        f.write(" | ")
                        f.write(c)
                        f.write("\n")
                    f.write(">> ")
                    for key,value in metrics.items():
                        f.write(str(key)+" : "+str(value)+" | ")
                    f.write("\n")
                    f.write("####################\n")

        else:
            if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, 
                data_info=self.get_data_info(metric_key_prefix),input_ids= all_input_ids), task)
            else:
                metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def verbalizer_prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    
                    loss, predictions, labels = self.verbalizer_compute_loss(model, inputs, return_outputs=True)
                    loss = loss.type(torch.FloatTensor)
                    loss = loss.mean().detach()

                    outputs = predictions

                else:
                    loss = None
                    outputs = model(**inputs)
                
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
    
        return (loss, outputs, labels)
    
    def verbalizer_compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(inputs)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        if "labels" in inputs:
            #labels = inputs.pop("labels")
            labels = inputs["labels"]
            labels[labels[:,:] == -100] = self.tokenizer.pad_token_id
        else:
            labels = None
        
        prob_list = []
        loss_list = []
        logits_list = []
        with torch.no_grad():
            transposed_labels_list = [list(x) for x in zip(*inputs['labels_list'])]
            
            for index in range(len(transposed_labels_list)):
                option = transposed_labels_list
                option_ = self.tokenizer.batch_encode_plus(option[index], max_length=self.data_args.val_max_target_length,
                                                        padding=True, truncation=True, return_tensors="pt")
                lm_labels = option_["input_ids"].expand(len(inputs['input_ids']),-1)
                lm_labels[lm_labels[:,:] == self.tokenizer.pad_token_id] = -100
                outputs = model(
                    input_ids = inputs['input_ids'].cuda(),
                    attention_mask = inputs['attention_mask'].cuda(),
                    labels=lm_labels.cuda(),
                    decoder_attention_mask = option_["attention_mask"].cuda(),
                    #task=inputs['task']
                )
                logits = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)
                lm_labels = lm_labels.cuda().unsqueeze(-1)
                seq_token_log_prob = torch.zeros(lm_labels.shape)
                
                for i in range(lm_labels.size(0)):
                    for j in range(lm_labels.size(1)):
                        seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                loss_list.append(outputs.loss)
                logits_list.append(logits)
                prob_list.append(seq_log_prob)
                
            concat = torch.cat(prob_list).view(-1,len(inputs['input_ids']))
            # TODO : Check if argmax or argmin
            prediction_indices = concat.argmax(dim=0)
            predictions = [inputs['labels_list'][elem_num][i.item()] for elem_num, i in enumerate(prediction_indices)]
            predictions = self.tokenizer.batch_encode_plus(predictions, max_length=self.data_args.val_max_target_length,
                                                        padding=True, truncation=True, return_tensors="pt")
            predictions = predictions['input_ids']
    
            loss = torch.mean(torch.stack(loss_list),dim=0)
            
        return (loss, predictions, labels) if return_outputs else loss
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.model = self.model.to(args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                load_result = self.model.load_state_dict(state_dict, strict=False)
                if len(load_result.missing_keys) != 0:
                    if load_result.missing_keys == self.model._keys_to_ignore_on_save:
                        self.model.tie_weights()
                    else:
                        logger.warn(
                            f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                        )
                if len(load_result.unexpected_keys) != 0:
                    logger.warn(
                        f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
                    )

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(args.device)
            self.model_wrapped = self.model


        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # # Train!
        # if is_torch_tpu_available():
        #     world_size = xm.xrt_world_size()
        # elif args.local_rank != -1:
        #     world_size = dist.get_world_size()
        # else:
        #     world_size = 1
        world_size=1

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            # if is_torch_tpu_available():
            #     xm.rendezvous("load_best_model_at_end")
            # elif args.local_rank != -1:
            #     dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # TODO :sanity check
        if len(self.train_dataset)>50000:
            g = torch.Generator()
            g.manual_seed(self.data_args.data_seed)
            rp = torch.randperm(len(self.train_dataset),generator=g).tolist()
            rp = rp[:50000]
            self.train_dataset = self.train_dataset.select(rp)

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            wandb.log({f"train_loss": logs['loss']})
            wandb.log({f"learning_rate": logs['learning_rate']})
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)
        
        if self.args.do_eval:
            metrics = None
            if self.control.should_evaluate:
                logger.info("*** Evaluate ***")
                for (task, eval_dataset), eval_config in zip(self.eval_datasets.items(),self.eval_data_configs):
                    print('@$%$%$%$%$%$%$%$$$$$$$$$$$$$$$$@')
                    print(task)
                    print()
                    print(eval_dataset)
                    print('@$%$%$%$%$%$%$%$$$$$$$$$$$$$$$$@')
                    metrics = self.evaluate(eval_dataset=eval_dataset,
                                            max_length=self.data_args.val_max_target_length, num_beams=self.data_args.num_beams,
                                            task = task)
                    #trainer.log_metrics(f"eval_{task}_", metrics)
                    #trainer.save_metrics(f"eval_{task}_", metrics)
                    wandb.log({f"eval_{task}_{eval_config}": metrics})
                    self._report_to_hp_search(trial, epoch, metrics)
                    
                #metrics = self.evaluate()
                #self._report_to_hp_search(trial, epoch, metrics)
            
            if self.control.should_save:
                self._save_checkpoint(model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)