import os
import argparse
import logging
# from mag.experiment import Experiment
# import mag
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model import get_model_optimizer
from loops import train_loop, evaluate, infer
from dataset import cross_validation_split, get_test_dataset, BucketingSampler, make_collate_fn
from transformers import BertTokenizer, AlbertTokenizer
from torch.utils.data import DataLoader, Dataset
from evaluation import target_metric
from misc import target_columns, input_columns


from args import parser

# parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataframe", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)

args = parser.parse_args()

logging.getLogger("transformers").setLevel(logging.ERROR)

test_df = pd.read_csv(args.dataframe)

tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=("uncased" in args.bert_model)
)

test_set = get_test_dataset(args, test_df, tokenizer)
test_loader = DataLoader(
    test_set,
    batch_sampler=BucketingSampler(
        test_set.lengths,
        batch_size=args.batch_size,
        maxlen=args.max_sequence_length
    ),
    collate_fn=make_collate_fn(),
)

os.makedirs(args.output_dir)

for fold in range(args.folds):

    print()
    print("Fold:", fold)
    print()

    fold_checkpoints = os.path.join(
        args.checkpoints_path, "fold{}".format(fold)
    )

    model, optimizer = get_model_optimizer(args)

    checkpoint = os.path.join(fold_checkpoints, args.checkpoint)

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()

    test_preds = infer(
        args, model, test_loader, test_shape=len(test_set)
    )

    del model, optimizer
    torch.cuda.empty_cache()

    test_preds_df = test_df[["qa_id"]].copy()
    for k, col in enumerate(target_columns):
        test_preds_df[col] = test_preds[:, k].astype(np.float32)
    test_preds_df.to_csv(
        os.path.join(args.output_dir, "fold-{}.csv".format(fold)),
        index=False,
    )

