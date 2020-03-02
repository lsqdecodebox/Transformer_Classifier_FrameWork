import warnings, logging
import gc
import random
import os, multiprocessing, glob
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from transformers import get_linear_schedule_with_warmup
from model import get_model_optimizer
from loops import train_loop, evaluate, infer
from dataset import cross_validation_split, get_test_dataset, get_pseudo_dataset, make_collate_fn, BucketingSampler
from args import args
from transformers import BertTokenizer, AlbertTokenizer
from torch.utils.data import DataLoader, Dataset


# lingo configuration
# args.bert_model = '../huggingface-bert-base-uncased-pytorch'
# args.is_cuda = False

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
seed_everything(args.seed)


## load the data
train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))

tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=("uncased" in args.bert_model)
)

test_dataset = get_test_dataset(args, test_df, tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=BucketingSampler(     # for 优化, sigma(per_sample_length) = batch size * max_length
        test_dataset.lengths,
        batch_size=args.batch_size,
        maxlen=args.max_sequence_length
    ),
    collate_fn=make_collate_fn(),
)


for fold, train_dataset, valid_dataset, train_fold_df, val_fold_df in (
        cross_validation_split(
            args,
            train_df,
            tokenizer
        )
):

    print()
    print("Fold:", fold)
    print()

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=BucketingSampler(
            valid_dataset.lengths,
            batch_size=args.batch_size,
            maxlen=args.max_sequence_length
        ),
        collate_fn=make_collate_fn(),
    )

    # 文件输出配置
    fold_checkpoints = os.path.join(
        args.checkpoints_path , "fold{}".format(fold)
    )
    fold_predictions = os.path.join(
        args.predictions_path, "fold{}".format(fold)
    )
    os.makedirs(fold_checkpoints, exist_ok=True)
    os.makedirs(fold_predictions, exist_ok=True)


    iteration = 0
    best_score = -1.0

    model, optimizer = get_model_optimizer(args)
    criterion = nn.BCEWithLogitsLoss()


    for epoch in range(args.epochs):

        epoch_train_dataset = train_dataset

        if args.pseudo_file is not None:

            pseudo_df = pd.read_csv(args.pseudo_file.format(fold))

            pseudo_set = get_pseudo_dataset(
                args,
                pseudo_df.sample(args.n_pseudo),
                tokenizer
            )
            epoch_train_dataset = ConcatDataset([epoch_train_dataset, pseudo_set])

        train_loader = DataLoader(   # train loader 不进行sample了？
            epoch_train_dataset,
            batch_size=args.batch_size,
            # num_workers=args.workers,
            collate_fn=make_collate_fn(),
            drop_last=True,
            shuffle=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=(
                args.epochs * len(train_loader) / args.batch_accumulation
            ),
        )

        avg_loss, iteration = train_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            args,
            iteration,
        )

        avg_val_loss, score, val_preds = evaluate(
            args,
            model,
            valid_loader,
            criterion,
            val_shape=len(valid_dataset)
        )

        print(
            "Epoch {}/{}: \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f}".format(
                epoch + 1, args.epochs, avg_loss, avg_val_loss, score
            )
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                fold_checkpoints, "model_on_epoch_{}.pth".format(epoch)
            ),
        )

        val_preds_df = val_fold_df.copy()[["qa_id"] + args.target_columns]
        val_preds_df[args.target_columns] = val_preds
        val_preds_df.to_csv(
            os.path.join(fold_predictions, "val_on_epoch_{}.csv".format(epoch)),
            index=False,
        )

        test_preds = infer(args, model, test_loader, test_shape=len(test_dataset))
        test_preds_df = submission.copy()
        test_preds_df[args.target_columns] = test_preds
        test_preds_df.to_csv(
            os.path.join(fold_predictions, "test_on_epoch_{}.csv".format(epoch)),
            index=False,
        )

        if score > best_score:
            best_score = score
            torch.save(
                model.state_dict(),
                os.path.join(fold_checkpoints, "best_model.pth"),
            )
            val_preds_df.to_csv(
                os.path.join(fold_predictions, "best_val.csv"), index=False
            )
            test_preds_df.to_csv(
                os.path.join(fold_predictions, "best_test.csv"), index=False
            )
    del model, optimizer, criterion, scheduler
    del valid_loader, train_loader, valid_dataset, train_dataset
    torch.cuda.empty_cache()
    gc.collect()

    print()


best_val_df_files = [
    os.path.join(args.predictions_path, "fold{}".format(fold), "best_val.csv")
    for fold in range(args.folds)
]

if all(os.path.isfile(file) for file in best_val_df_files):
    best_val_dfs = [pd.read_csv(file) for file in best_val_df_files]
    oof_df = pd.concat(best_val_dfs).reset_index(drop=True)
    oof_df.to_csv(os.path.join(args.predictions_path, "oof.csv"), index=False)
