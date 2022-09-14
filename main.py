import datetime
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoConfig, \
    AutoModelForSequenceClassification
from transformers import (
    HfArgumentParser)
from transformers.trainer_utils import get_last_checkpoint, set_seed

from MyTrainer import CustomTrainer
from data_helper import read_data
from model import RobertaCRF

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='roberta-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default='cache/', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: str = field(
        default='raw_data/qg_train.json',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        default='raw_data/qg_valid.json',
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    task: Optional[str] = field(
        default='paragraph_selection',
        metadata={
            "help": "cloze2normal, normal2cloze, multi, qg"},
    )

    answer_aware: Optional[int] = field(
        default=0,
        metadata={"help": 'include answer during training?'},
    )

    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=48,
        metadata={"help": "Max input length for the target text"},
    )

    is_debug_mode: Optional[int] = field(
        default=1,
        metadata={"help": "training on local machine?"},
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    # crf: Optional[int] = field(
    #     default=1,
    #     metadata={
    #         "help": ("content selection as a sequence labeling problem")
    #     },
    # )


def set_loggers(training_args):
    time = datetime.datetime.now()

    output_dir = "./results/results_{}_{}_{}_{}_ep{}_lr{}_seed{}".format(
        # training_args.task,
        time.month,
        time.day,
        time.hour,
        time.minute,
        # args.experiment,
        # args.model_name.split('/')[0],
        str(training_args.num_train_epochs),
        str(training_args.learning_rate),
        str(training_args.seed),
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed
    set_seed(training_args.seed)
    # Set project name
    os.environ["WANDB_PROJECT"] = "para_selection"

    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    log_dir = output_dir + '/logs'
    return output_dir, log_dir


import numpy as np


#
# def compute_metrics(pred):
#     # *_, ls = pred.label_ids.shape
#     pred_label = pred.label_ids.flatten()
#     pred_pred = pred.predictions.flatten()
#     labels = np.delete(pred_label, np.where(pred_label == -100))
#     preds = np.delete(pred_pred, np.where(pred_pred == -100))
#     # labels = np.reshape(pred.label_ids[np.where(pred.label_ids != -100)], (-1, ls))
#     # preds = np.reshape(pred.predictions[np.where(pred.predictions != -100)], (-1, ls))
#     # labels = [item[np.where(item != -100)] for item in pred.label_ids]
#     # preds = [item[np.where(item != -100)] for item in pred.predictions]
#     # precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(labels, preds)
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
#     precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     #
#     #     print(classification_report(labels, preds, labels=le.transform(le.classes_), target_names=le.classes_.tolist()))
#     #     precision_s, recall_s, f1_s = 0., 0., 0.
#     # else:
#     #     precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(labels, preds, average='samples')
#
#     return {
#         # 'precision_sampled': precision_s,
#         # 'recall_sampled': recall_s,
#         # 'f1_sampled': f1_s,
#
#         'precision_micro': precision_micro,
#         'recall_micro': recall_micro,
#         'f1_micro': f1_micro,
#
#         'precision_macro': precision_macro,
#         'recall_macro': recall_macro,
#         'f1_macro': f1_macro,
#
#         'precision_weighted': precision_w,
#         'recall_weighted': recall_w,
#         'f1_weighted': f1_w,
#
#         'accuracy': accuracy_score(labels, preds)
#     }


def compute_metrics(pred):
    labels = pred.label_ids
    # preds = (pred.predictions > 0.5).astype(int)
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
    }


def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # set seed & init logger
    output_dir, logdir = set_loggers(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        model_max_length=512,
        # use_auth_token=True if model_args.use_auth_token else None,
    )

    # ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['[LABEL_0]', '[LABEL_1]'], 'pad_token': '[PAD]'}
    # orig_num_tokens = len(tokenizer)
    # num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)

    train_ds, _, valid_ds, valid_raw_data = read_data(data_args, tokenizer)
    if data_args.is_debug_mode == 1:
        print('tokenization finished...')
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = 2
        config.num_hidden_layers = 2
        config.hidden_size = 32
        # config.intermediate_size = 128
        config.num_attention_heads = 2
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                   cache_dir=model_args.cache_dir,
                                                                   num_labels=2)

    # if num_added_tokens > 0:
    #     model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    # model = RobertaCRF(encoder, data_args.crf)

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=training_args.num_train_epochs,  # total number of training epochs
        learning_rate=training_args.learning_rate,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logdir,  # directory for storing logs
        # load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        # logging_steps=300,  # log & save weights each logging_steps
        # save_steps=300,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        save_strategy="no",
        evaluation_strategy="epoch",  # evaluate each `logging_steps`
    )

    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = CustomTrainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        tokenizer=tokenizer,
        train_dataset=train_ds,  # training dataset
        eval_dataset=valid_ds,  # evaluation dataset
        # data_collator=data_collator,
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )

    trainer.train()
    # trainer.evaluate()

    ######################################################
    ######################################################
    # save results on json file
    # np.delete(pred_pred, np.where(pred_pred == -100))
    from itertools import chain
    # with open(output_dir + '/{}_eval_scores.json'.format(output_dir.split('/')[-1]), 'w') as outfile:
    # save results on json file
    with open(output_dir + '/{}_eval_scores.json'.format(output_dir.split('/')[-1]), 'w') as outfile:
        # pred_train = trainer.predict(train_ds)
        pred_valid = trainer.predict(valid_ds)
        # pred_train = [[t[0].tolist()[0], t[1].tolist()] for t in zip(pred_train.predictions, pred_train.label_ids)]
        pred_valid = [t.tolist() for t in pred_valid.predictions]

        # train_txt = tokenizer.batch_decode([item['input_ids'] for item in train_ds],
        #                                    skip_special_tokens=True,
        #                                    clean_up_tokenization_spaces=True)

        # valid_txt = tokenizer.batch_decode([item['input_ids'] for item in valid_ds],
        #                                    skip_special_tokens=True,
        #                                    clean_up_tokenization_spaces=True)

        tmp = {
            # 'train_pred':
            #     [(item[0], item[1]) for item in zip(train_txt, pred_train)],
            # do not save augmented
            'valid_pred':
                [(item[0], item[1]) for item in zip(valid_raw_data, pred_valid)],
        }
        json.dump(tmp, outfile)


if __name__ == '__main__':
    main()
