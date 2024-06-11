import pandas as pd
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from os.path import join
import json
import re
from fuzzywuzzy import fuzz
from collections import defaultdict as dd
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import copy
import sentence_transformers
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CECorrelationEvaluator,
    CEBinaryClassificationEvaluator,
    CEBinaryAccuracyEvaluator,
)

from sentence_transformers import InputExample
from torch.utils.data import DataLoader

import math
from collections import OrderedDict

from sklearn.model_selection import KFold, GroupKFold

from util import seed_torch, load_json, dump_json, sigmoid
import unicodedata
import nltk
import string

nltk.download("stopwords")

import dataclasses


def str_normalize(s):
    # URL削除、カタカナ英数字の半角全角正規化、絵文字削除
    norm_text = re.sub(r"(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?", "", s)
    norm_text = re.sub(
        r'<div>|</div>|<profileDesc>|</profileDesc>|<teiHeader>|</teiHeader>|<abstract>|</absstract>|<p>|</p>|<ul>|</ul>|<li>|</li>|<a href="">|</a>|<code>|</code>|<blockquote>|</blockquote>|<a.*/a>|\n|&quot;',
        "",
        norm_text,
    )
    norm_text = re.sub(r"<ref[^>]*>(.*?)<\/ref>", "", norm_text)
    norm_text = re.sub(r"<head[^>]*>(.*?)<\/head>", "", norm_text)
    norm_text = re.sub(r"<div[^>]*>(.*?)<\/div>", "", norm_text)
    norm_text = norm_text.replace("\t", "")
    norm_text = norm_text.replace(".", " ")

    norm_text = norm_text.lower()

    # reference https://github.com/jbesomi/texthero/blob/master/texthero/preprocessing.py
    ## remove_diacritics 発音区別符号 や アクセント を削除
    nfkd_form = unicodedata.normalize("NFKD", norm_text)
    norm_text = "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    ## remove_whitespace
    norm_text = norm_text.replace("\xa0", " ")

    ## replace_punctuation
    norm_text = norm_text.replace(rf"([{string.punctuation}])+", "")

    ## remove stopwords
    from nltk.corpus import stopwords as nltk_en_stopwords

    words = set(nltk_en_stopwords.words("english"))

    pattern = r"""(?x)                          # Set flag to allow verbose regexps
      \w+(?:-\w+)*                              # Words with optional internal hyphens
      | \s*                                     # Any space
      | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol
    """
    norm_text = "".join(
        t if t not in words else "" for t in re.findall(pattern, norm_text)
    )

    return norm_text


def row_tiltle_context(row, context_col="context"):
    return [
        str(row["ref_title"]).lower(),
        str(row["title"]).lower() + str(row[context_col]).lower(),
    ]


def row_ref_abst(row, context_col="context"):
    return [
        str(row[context_col]).lower()
        + " "
        + str(row["ref_title"]).lower()
        + " "
        + str_normalize(str(row["ref_abstract"])).lower(),
        str(row["title"]).lower() + " " + str_normalize(str(row["abstract"])).lower(),
    ]


# def row_intro_ref_abst(row, context_col="context"):
#     return [
#         str(row[context_col]).lower()
#         + " "
#         + str(row["ref_title"]).lower()
#         + " "
#         + str_normalize(str(row["ref_abstract"])).lower(),
#         str(row["title"]).lower()
#         + " "
#         + str_normalize(str(row["abstract"])).lower()
#         + " "
#         + str_normalize(str(row["introduction"])).lower(),
#     ]


def row_intro_ref_abst(row, context_col="context"):
    return [
        (
            "referenced in the follewing text "
            + str(row[context_col]).lower()
            + ", title "
            + str(row["ref_title"]).lower()
            + ", abstract "
            + str_normalize(str(row["ref_abstract"])).lower()
            + ", keywords "
            + str(" ".join(eval(row["ref_keywords"]))).lower()
            if type(row["ref_keywords"]) != float
            else ""
        ),
        (
            "title "
            + str(row["title"]).lower()
            + ", abstract "
            + str_normalize(str(row["abstract"])).lower()
            + ", introduction "
            + str_normalize(str(row["introduction"])).lower()
            + " , keywords "
            + str(" ".join(eval(row["keywords"]))).lower()
            if type(row["keywords"]) != float
            else ""
        ),
    ]


def row_raw(row, context_col="context"):
    return [
        (
            str(row[context_col]).lower()
            + " "
            + str(row["ref_title"]).lower()
            + " "
            + str(row["ref_abstract"]).lower()
            + " "
            + str(" ".join(eval(row["ref_keywords"]))).lower()
            if type(row["ref_keywords"]) != float
            else ""
        ),
        (
            str(row["title"]).lower()
            + " "
            + str(row["abstract"]).lower()
            + " "
            + str(" ".join(eval(row["keywords"]))).lower()
            if type(row["keywords"]) != float
            else "" + " " + str(row["introduction"]).lower()
        ),
    ]


def row_abstract(row, context_col="context"):
    return [
        str(row[context_col]).lower() + " " + str(row["ref_title"]).lower(),
        str(row["title"]).lower() + " " + str_normalize(str(row["abstract"])).lower(),
    ]


def row_introduction(row, context_col="context"):
    return [
        str(row[context_col]).lower() + " " + str(row["ref_title"]).lower(),
        str(row["title"]).lower()
        + " "
        + str_normalize(str(row["introduction"])).lower(),
    ]


def row_default(row, context_col="context"):
    return [
        str(row[context_col]).lower() + " " + str(row["ref_title"]).lower(),
        str(row["title"]).lower(),
    ]


def row_tile_ref_title(row, context_col="context"):
    return [str(row["ref_title"]).lower(), str(row["title"]).lower()]


def row_only_title_context(row, context_col="context"):
    return [str(row[context_col]).lower(), str(row["title"]).lower()]


def create_samples(df_context, params, filter_empty_context=False):
    samples = []

    for i, row in df_context.iterrows():

        if (
            filter_empty_context
            and str(row[params.context_col]) == ""
            and str(row["ref_title"]) == ""
        ):
            continue

        if filter_empty_context and str(row["title"]) == "":
            continue

        if params.context_mode == "abstract":
            samples.append(
                InputExample(
                    texts=row_abstract(row, context_col=params.context_col),
                    label=row["label"],
                )
            )
        elif params.context_mode == "introduction":
            samples.append(
                InputExample(
                    texts=row_introduction(row, context_col=params.context_col),
                    label=row["label"],
                )
            )

        elif params.context_mode == "title_context":
            samples.append(
                InputExample(
                    texts=row_tiltle_context(row, context_col=params.context_col),
                    label=row["label"],
                )
            )

        elif params.context_mode == "ref_abst":
            samples.append(
                InputExample(
                    texts=row_ref_abst(row, context_col=params.context_col),
                    label=row["label"],
                )
            )

        elif params.context_mode == "title_ref_title":
            samples.append(
                InputExample(
                    texts=row_tile_ref_title(row, context_col=params.context_col),
                    label=row["label"],
                )
            )
        elif params.context_mode == "intro_ref_abst":
            samples.append(
                InputExample(
                    texts=row_intro_ref_abst(row, context_col=params.context_col),
                    label=row["label"],
                )
            )
        elif params.context_mode == "row":

            samples.append(
                InputExample(
                    texts=row_raw(row, context_col=params.context_col),
                    label=row["label"],
                )
            )
        elif params.context_mode == "only_title_context":
            samples.append(
                InputExample(
                    texts=row_only_title_context(row, context_col=params.context_col),
                    label=row["label"],
                )
            )
        else:
            samples.append(
                InputExample(
                    texts=row_default(row, context_col=params.context_col),
                    label=row["label"],
                )
            )

    return samples


def get_surrounding_numbers(n, m):
    """
    指定された数値の前後m個の数値を含むリストを返します。
    リストの要素数はmとなります。負の数値は結果に含めません。

    :param n: 基準となる正の整数
    :param m: リストの要素数
    :return: nを中心とした前後m//2個の数値を含むリスト
    """
    if m % 2 == 0:
        # mが偶数の場合、中心となるnの前後にm//2個の要素を取る
        start = max(0, n - (m // 2))
        end = start + m
    else:
        # mが奇数の場合、中心となるnの前にm//2個、後ろにm//2個を取る
        start = max(0, n - (m // 2))
        end = start + m

    # リストの要素がm個になるように調整
    if end <= n + (m // 2):
        end = n + (m // 2) + 1
        start = end - m
        start = max(0, start)

    return [i for i in range(start, end)]


def create_samples_negative_sample(
    df_context: pd.DataFrame,
    n_samples: int,
    params,
    filter_empty_context: bool = False,
    random_state=42,
    # abstract: bool = False,
    # introduction: bool = False,
    # title_context: bool = False,
    # ref_abst: bool = False,
):
    """
    create samples with negative samples
    if n_samples == 0, same number of negative samples as positive samples
    """

    assert n_samples >= 0

    samples = []

    for pid, df in df_context.groupby("pid"):

        df_pos = df[df["label"] == 1]

        if n_samples == 0:
            n_samples = len(df_pos)

        if params.neg_sample_mode == "random":
            df_neg = df[df["label"] == 0].sample(
                n_samples, random_state=random_state, replace=True
            )
        elif params.neg_sample_mode == "surrounding":
            df_neg = df[df["label"] == 0]
            df_neg["bid_int"] = df_neg["bid"].apply(lambda x: int(x[1:]))

            _bids = []
            for pos_bid in df_pos["bid"].tolist():
                pos_bid_int = int(pos_bid[1:])
                surrounding_bids = get_surrounding_numbers(
                    pos_bid_int, n_samples // len(df_pos)
                )
                _bids.extend(surrounding_bids)

            df_neg = df_neg[df_neg["bid_int"].isin(_bids)]

        # positive
        for i, row in df_pos.iterrows():

            if (
                filter_empty_context
                and str(row["context"]) == ""
                and str(row["ref_title"]) == ""
            ):
                continue

            if filter_empty_context and str(row["title"]) == "":
                continue

            if params.context_mode == "abstract":
                samples.append(
                    InputExample(
                        texts=row_abstract(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "introduction":
                samples.append(
                    InputExample(
                        texts=row_introduction(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "title_context":
                samples.append(
                    InputExample(
                        texts=row_tiltle_context(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "ref_abst":
                samples.append(
                    InputExample(
                        texts=row_ref_abst(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "title_ref_title":
                samples.append(
                    InputExample(
                        texts=row_tile_ref_title(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "intro_ref_abst":
                samples.append(
                    InputExample(
                        texts=row_intro_ref_abst(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "row":
                samples.append(
                    InputExample(
                        texts=row_raw(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "only_title_context":
                samples.append(
                    InputExample(
                        texts=row_only_title_context(row),
                        label=row["label"],
                    )
                )

            else:
                samples.append(
                    InputExample(
                        texts=row_default(row),
                        label=row["label"],
                    )
                )

        # negative
        for i, row in df_neg.iterrows():

            if (
                filter_empty_context
                and str(row["context"]) == ""
                and str(row["ref_title"]) == ""
            ):
                continue

            if filter_empty_context and str(row["title"]) == "":
                continue

            if params.context_mode == "abstract":
                samples.append(
                    InputExample(
                        texts=row_abstract(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "introduction":
                samples.append(
                    InputExample(
                        texts=row_introduction(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "title_context":
                samples.append(
                    InputExample(
                        texts=row_tiltle_context(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "ref_abst":
                samples.append(
                    InputExample(
                        texts=row_ref_abst(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "title_ref_title":
                samples.append(
                    InputExample(
                        texts=row_tile_ref_title(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "intro_ref_abst":
                samples.append(
                    InputExample(
                        texts=row_intro_ref_abst(row),
                        label=row["label"],
                    )
                )
            elif params.context_mode == "row":
                samples.append(
                    InputExample(
                        texts=row_raw(row),
                        label=row["label"],
                    )
                )

            elif params.context_mode == "only_title_context":
                samples.append(
                    InputExample(
                        texts=row_only_title_context(row),
                        label=row["label"],
                    )
                )

            else:
                samples.append(
                    InputExample(
                        texts=row_default(row),
                        label=row["label"],
                    )
                )

    return samples


def create_data_loader(df_context, batch_size=32, shuffle=True, params=None):
    if params.n_negative_sample == -1:
        print(params)
        samples = create_samples(df_context, params, filter_empty_context=True)
    else:
        samples = create_samples_negative_sample(
            df_context,
            params.n_negative_sample,
            params,
            filter_empty_context=True,
            random_state=params.seed,
        )
    seed_torch(2024)

    print("sample text")
    print(samples[0].texts)

    return DataLoader(samples, shuffle=shuffle, batch_size=batch_size)


def create_test_data(df_context, params):
    if params.context_mode == "abstract":
        return [row_abstract(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "introduction":
        return [row_introduction(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "title_context":
        return [row_tiltle_context(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "ref_abst":
        return [row_ref_abst(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "title_ref_title":
        return [row_tile_ref_title(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "intro_ref_abst":
        return [row_intro_ref_abst(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "row":
        return [row_raw(row) for i, row in df_context.iterrows()]
    elif params.context_mode == "only_title_context":
        return [row_only_title_context(row) for i, row in df_context.iterrows()]
    else:
        return [row_default(row) for i, row in df_context.iterrows()]


def ap(actual, predicted):

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(actual)


def evaluate_mapk(true, pred, df_valid_pid):

    df_valid_pid["label"] = true
    df_valid_pid["pred"] = pred

    ap_list = []
    pid_list = []

    for pid, df in tqdm(df_valid_pid.groupby("pid")):
        df = df.reset_index().rename(columns={"index": "rid"})
        truth = df[df["label"] == 1]["rid"].sort_values().tolist()
        pred = df[["rid", "pred"]].sort_values("pred", ascending=False)["rid"].tolist()
        _ap = ap(truth, pred)
        ap_list.append(_ap)
        pid_list.append(pid)

    df_ap = pd.DataFrame({"pid": pid_list, "ap": ap_list})

    return np.mean(ap_list), df_ap


def train_k_fold(
    df_context: pd.DataFrame,
    params,
):
    print("model name", params.model_name)

    if params.debug:
        pid = df_context["pid"].sample(50, random_state=2024)
        df_context = df_context[df_context["pid"].isin(pid)].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif params.model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
    elif params.model_name == "scideberta":
        BERT_MODEL = "KISTI-AI/scideberta"
    elif params.model_name == "deberta":
        BERT_MODEL = "microsoft/deberta-v3-base"
    elif params.model_name == "distilbert":
        BERT_MODEL = "distilbert/distilbert-base-uncased"
    elif params.model_name == "roberta":
        BERT_MODEL = "facebook/muppet-roberta-base"
    elif params.model_name == "xlnet":
        BERT_MODEL = "xlnet/xlnet-base-cased"
    elif params.model_name == "albert":
        BERT_MODEL = "albert/albert-base-v2"
    elif params.model_name == "scideberta_full":
        BERT_MODEL = "KISTI-AI/Scideberta-full"
    else:
        print(f"unkown model name: {params.model_name}")
        raise NotImplementedError

    BATCH_SIZE = params.batch_size
    group_kfold = GroupKFold(n_splits=params.k)

    map_list = []
    df_oof_pred = df_context[["pid", "bid"]].copy()
    df_oof_pred["pred"] = 0.0
    ap_list = []

    noise_paper = ["557d11a66feeaa8086da604a"]

    for n_fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(df_context, groups=df_context["pid"])
    ):

        print("-------------------- " + str(n_fold) + " --------------------")

        model = CrossEncoder(BERT_MODEL, num_labels=1, max_length=params.MAX_SEQ_LENGTH)

        df_context_train = df_context.iloc[train_idx].reset_index()
        df_context_train = df_context_train[~df_context_train["pid"].isin(noise_paper)]

        train_group_list = df_context_train["pid"].unique().tolist()
        split = int(len(train_group_list) * 0.8)

        df_context_valid = df_context_train[
            df_context_train["pid"].isin(train_group_list[split:])
        ]
        df_context_train = df_context_train[
            df_context_train["pid"].isin(train_group_list[:split])
        ]

        df_context_test = df_context.iloc[test_idx].reset_index()

        print("prepare train dataloader")
        train_dataloader = create_data_loader(
            df_context_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            params=params,
        )

        print("prepare evaluator")
        dev_samples = create_samples(df_context_valid, params)
        evaluator = CECorrelationEvaluator.from_input_examples(
            dev_samples, name="sts-dev"
        )
        # evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        #    dev_samples, name="sts-dev"
        # )

        warmup_steps = math.ceil(
            len(train_dataloader) * params.epoch * 0.1
        )  # 10% of train data for warm-up
        print("Warmup-steps: {}".format(warmup_steps))

        if params.ubm:
            if params.n_negative_sample == -1:
                # pos_weight = torch.tensor([30.0]).to(device)
                # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

            model.fit(
                train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=params.epoch,
                warmup_steps=warmup_steps,
                output_path=params.model_save_path + "/model_" + str(n_fold),
                evaluation_steps=200,
                save_best_model=True,
                loss_fct=criterion,
            )

            best_model = CrossEncoder(
                params.model_save_path + "/model_" + str(n_fold),
                max_length=params.MAX_SEQ_LENGTH,
            )
        else:
            model.fit(
                train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=params.epoch,
                warmup_steps=warmup_steps,
                output_path=params.model_save_path + "/model_" + str(n_fold),
            )

            best_model = model

        pred = best_model.predict(
            create_test_data(df_context_test, params),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        )

        mapk, df_ap = evaluate_mapk(
            df_context_test["label"], pred, df_context_test[["pid"]].copy()
        )

        print("map: ", mapk)
        map_list.append(mapk)
        ap_list.append(df_ap)
        df_oof_pred.loc[test_idx, ("pred",)] = pred

    df_ap = pd.concat(ap_list)

    return df_oof_pred, map_list, df_ap


def predict(
    df_test_context,
    params,
    model_save_path="model",
    k=5,
    batch_size=4,
):

    df_pred = df_test_context[["pid", "bid"]].copy()
    test_dataset = create_test_data(df_test_context, params)
    for n_fold in range(k):
        model = CrossEncoder(
            model_save_path + "/model_" + str(n_fold),
            num_labels=1,
            max_length=params.MAX_SEQ_LENGTH,
        )

        pred = model.predict(
            test_dataset,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        df_pred[f"score_fold_{n_fold}"] = pred

    df_pred["score_mean"] = df_pred[[f"score_fold_{i}" for i in range(k)]].mean(axis=1)

    return df_pred


def df2sub_dict(df_pred):
    sub_dict = {}
    for pid, df in df_pred.groupby("pid"):
        df["bid_int"] = df["bid"].apply(lambda x: int(x[1:]))
        df = df[["bid_int", "score_mean"]].sort_values("bid_int").reset_index(drop=True)
        sub_dict[pid] = df["score_mean"].to_list()
    return sub_dict


@dataclasses.dataclass(frozen=True)
class Params:
    epoch: int
    model_name: str
    model_save_path: str
    k: int
    debug: bool
    n_negative_sample: int
    context_mode: str
    MAX_SEQ_LENGTH: int = 512
    ubm: bool = False
    batch_size: int = 16
    # title_context: bool = False
    # ref_abst: bool = False
    # abstract: bool
    # introduction: bool
    seed: int = 2024
    neg_sample_mode: str = "random"
    context_col: str = "context"


def parse_arg():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, default="scibert", help="default scibert"
    )
    parser.add_argument("--epoch", type=int, default=1, help="default 1")
    parser.add_argument("--dplp_path", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--model_save_path", type=str, default="model")
    # parser.add_argument("--data_dir", type=str, default="../dataset/PST/")
    parser.add_argument("--k", type=int, default=5, help="default 5")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true")
    # parser.add_argument("--train_split", action="store_true")
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument(
        "--n_negative_sample", type=int, default=-1, help="-1 all negative. default -1"
    )
    # parser.add_argument(
    #     "--abstract", action="store_true", help="use abstract as context"
    # )
    # parser.add_argument(
    #     "--introduction", action="store_true", help="use introduction as context"
    # )

    # parser.add_argument(
    #     "--title_context", action="store_true", help="run title + context"
    # )

    # parser.add_argument("--ref_abst", action="store_true", help="run ref_abst")

    parser.add_argument(
        "--context_mode",
        choices=(
            "default",
            "abstract",
            "introduction",
            "title_context",
            "ref_abst",
            "title_ref_title",
            "intro_ref_abst",
            "row",
            "only_title_context",
        ),
        default="default",
    )

    parser.add_argument("--ubm", action="store_true", help="use best model")
    parser.add_argument("--seed", type=int, default=42, help="seed default 42")

    parser.add_argument(
        "--neg_sample_mode",
        type=str,
        default="random",
        choices=("random", "surrounding"),
        help="random or surrounding",
    )

    parser.add_argument(
        "--context_col",
        type=str,
        default="context",
        choices=("context", "context_clean"),
    )
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=512)

    # parser.add_argument("--neg_sample_mode", type=str, default="random")
    # parser.add_argument("--two_phase_learning", action="store_true")

    args = parser.parse_args()
    print("==== params =====")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    print("==== params =====")

    return args


def main():
    seed_torch(2024)
    args = parse_arg()

    params = Params(
        epoch=args.epoch,
        model_name=args.model_name,
        model_save_path=f"{args.output_dir}/{args.model_name}",
        k=args.k,
        debug=args.debug,
        n_negative_sample=args.n_negative_sample,
        context_mode=args.context_mode,
        ubm=args.ubm,
        batch_size=args.batch_size,
        seed=args.seed,
        neg_sample_mode=args.neg_sample_mode,
        context_col=args.context_col,
        MAX_SEQ_LENGTH=args.MAX_SEQ_LENGTH,
    )
    train_context_path = "../data/train_context.csv"
    # test_context_path = "data/test_pub_context_gen_title.csv"
    test_context_path = "../data/test_pub_gen_context_filled_citation.csv"

    if args.train:
        df_train_context = pd.read_csv(train_context_path)

        os.makedirs(args.output_dir, exist_ok=True)

        df_oof_pred, map_list, df_ap = train_k_fold(df_train_context, params)

        print(map_list)
        print(np.mean(map_list))

        df_ap.to_csv(f"{args.output_dir}/df_ce_ap_{args.model_name}.csv", index=None)

        dump_json(
            {"map": np.mean(map_list), "map_list": map_list},
            args.output_dir,
            "score.json",
        )

        df_oof_pred.to_csv(
            f"{args.output_dir}/oof_pred_{args.model_name}.csv", index=None
        )

    if args.prediction:

        df_test_context = pd.read_csv(test_context_path)

        df_test_score_all = predict(
            df_test_context,
            params,
            model_save_path=f"{args.output_dir}/{args.model_name}",
        )
        df_test_score_all.to_csv(
            f"{args.output_dir}/df_test_pub_score_all_{args.model_name}.csv", index=None
        )

        sub_dict = df2sub_dict(df_test_score_all)

        dump_json(
            sub_dict, args.output_dir, f"submission_{args.model_name}_test_pub.json"
        )


if __name__ == "__main__":
    main()
