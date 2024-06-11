import pandas as pd
import lightgbm
import numpy as np
import math
import os
import yaml

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from typing import List, Dict

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import LGBMRanker

from tqdm import tqdm

import json
from bs4 import BeautifulSoup
import optuna

import xgboost as xgb
from catboost import CatBoostClassifier, Pool, CatBoostRanker
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import SMOTE


from os.path import join


def dump_json(obj, wfdir, wfname):
    print("dumping %s ...", wfname)
    with open(join(wfdir, wfname), "w", encoding="utf-8") as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    print("%s dumped.", wfname)


def load_json(rfdir, rfname):
    print("loading %s ...", rfname)
    with open(join(rfdir, rfname), "r", encoding="utf-8") as rf:
        data = json.load(rf)
        print("%s loaded", rfname)
        return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_sigmoid(pred):
    new_pred = {}
    for k, v in pred.items():
        new_pred[k] = [sigmoid(x) for x in v]
    return new_pred


def fill_missing_group_mean(df):

    for col in df.columns:
        if (
            df[col].isnull().sum() > 0
            and df[col].dtype != "object"
            and col != "pid"
            and col != "bid"
            and col != "ref_id"
            and col != "ref_id"
        ):
            df[col] = df.groupby("pid")[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(0)

    return df


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
    for pid, df in tqdm(df_valid_pid.groupby("pid")):
        df = df.reset_index().rename(columns={"index": "rid"})
        truth = df[df["label"] == 1]["rid"].sort_values().tolist()
        pred = df[["rid", "pred"]].sort_values("pred", ascending=False)["rid"].tolist()
        if len(truth) == 0:
            continue
        _ap = ap(truth, pred)
        ap_list.append(_ap)

    return np.mean(ap_list)


def optuna_k_fold(
    X,
    y,
    group,
    clean_index,
    seed=42,
    n_splits=5,
    model_name="lightgbm",
    scale_pos_weight=25,
    parameter=[],
):

    oof_preds = np.zeros(X.shape[0])
    # folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = GroupKFold(n_splits=n_splits)
    train_indices = folds.split(X, y, group)
    best_params = []

    assert model_name in [
        "lightgbm",
        "catboost",
        "lightgbm_binary",
        "catboost_ranker",
        "logistic_regression",
    ]

    for n_fold, (train_idx, test_idx) in enumerate(train_indices):

        print("-------------------- " + str(n_fold) + " --------------------")

        # train_x, train_y, train_group = (
        #     X.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        #     y.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        #     group.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        # )

        train_x, train_y, train_group = (
            X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            group.iloc[train_idx].reset_index(drop=True),
        )

        test_x, test_y, test_group = (
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True),
            group.iloc[test_idx].reset_index(drop=True),
        )

        gkf_inner = GroupKFold(n_splits=4)
        train_inner_idx, valid_idx = next(
            gkf_inner.split(train_x, train_y, train_group)
        )

        val_x, val_y, valid_group = (
            train_x.iloc[valid_idx].reset_index(drop=True),
            train_y.iloc[valid_idx].reset_index(drop=True),
            train_group.iloc[valid_idx].reset_index(drop=True),
        )
        train_x, train_y, train_group = (
            train_x.iloc[train_inner_idx].reset_index(drop=True),
            train_y.iloc[train_inner_idx].reset_index(drop=True),
            train_group.iloc[train_inner_idx].reset_index(drop=True),
        )

        print(
            "train_x",
            train_x.shape,
            "train_y",
            train_y.shape,
            "train_group",
            train_group.shape,
        )
        print(
            "val_x", val_x.shape, "val_y", val_y.shape, "valid_group", valid_group.shape
        )

        # _train_group = train_group.value_counts().sort_index().values
        # _valid_group = valid_group.value_counts().sort_index().values

        print("Hyper-parameter Optimization")

        def objective_catboost(trial):
            params = {
                "random_seed": seed,
                "use_best_model": True,
                "eval_metric": "CrossEntropy",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 1, 10),
                #'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                #'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1),
                "learning_rate": 0.005,
                "random_strength": trial.suggest_int("random_strength", 0, 100),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.01, 100.00
                ),
                "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                "od_wait": trial.suggest_int("od_wait", 10, 50),
            }

            model = CatBoostClassifier(**params)
            train_pool = Pool(data=train_x, label=train_y)
            valid_pool = Pool(data=val_x, label=val_y)

            model.fit(
                train_pool, eval_set=valid_pool, use_best_model=True, verbose=False
            )
            val_pred = model.predict_proba(val_x)[:, 1]

            # auc = roc_auc_score(test_y, test_pred)
            loss = log_loss(val_y, val_pred)
            # mapk = evaluate_mapk(val_y, val_pred, pd.DataFrame({"pid": valid_group}))

            return loss

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective_catboost, n_trials=20)

        print(study.best_trial)

        best_params.append(study.best_params)

    return best_params


class SkPipeline:
    def __init__(self, model, predict_func_name="predict_proba"):
        self.stdsc = StandardScaler()
        self.model = model
        self.predict_func_name = predict_func_name

    def fit(self, X, y, X_valid, y_valid):
        # 標準化
        self.X_std = self.stdsc.fit_transform(X)
        # self.X_valid_std = self.stdsc.transform(X_valid)
        # 学習
        self.model.fit(
            self.X_std,
            y,
            # eval_set=[(self.X_valid_std, y_valid)],
            # eval_name = ["valid"],
            # eval_metric={'auc'},
            # max_epochs=5000,
            # from_unsupervised=self.pretrainer
        )
        return self

    def predict(self, X):
        self.X_std = self.stdsc.transform(X)
        if self.predict_func_name == "predict_proba":
            prediction = self.model.predict_proba(self.X_std)[:, 1]
        else:
            prediction = self.model.predict(self.X_std)
        return np.array(prediction)

    @property
    def coef_(self):
        return self.model.coef_


def run_under_sampling(df, seed=42, neg_scale=5):
    dfs = []
    for pid, _df in tqdm(df.groupby("pid"), total=df["pid"].nunique()):
        df_pos = _df[_df["label"] == 1].reset_index(drop=True)
        n = df_pos.shape[0]
        if n == 0:
            continue
        df_neg = _df[_df["label"] == 0].reset_index(drop=True)
        sample_num = n * neg_scale
        if df_neg.shape[0] > sample_num:
            df_neg = df_neg.sample(n=sample_num, random_state=seed)
        df = pd.concat([df, df_neg], axis=0).reset_index(drop=True)
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def train_k_fold(
    X,
    y,
    cols,
    group,
    clean_index,
    seed=42,
    n_splits=5,
    model_name="lightgbm",
    scale_pos_weight=25,
    parameter=[],
    under_sampling=False,
):
    """
    交差検定を行い、モデルを学習する
    """

    oof_preds = np.zeros(X.shape[0])
    # folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = GroupKFold(n_splits=n_splits)
    # folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # train_indices = folds.split(X, y)
    train_indices = folds.split(X, y, group)
    train_preds = np.zeros(X.shape[0])
    imps_list = []
    models = []
    map_score = []
    train_map_score = []
    auc_score = []
    train_auc_score = []
    valid_map_score = []
    valid_auc_score = []

    # assert model_name in [
    #     "lightgbm",
    #     "catboost",
    #     "lightgbm_binary",
    #     "catboost_ranker",
    #     "svm",
    #     "logistic_regression",
    #     "random_forest",
    # ]

    for n_fold, (train_idx, test_idx) in enumerate(train_indices):
        if model_name == "lightgbm":
            model = LGBMRanker(seed=seed, verbose=-1, n_estimators=1000)
        elif model_name == "lightgbm_binary":
            if len(parameter) > 0 and len(parameter[n_fold]) > 0:
                model = LGBMClassifier(seed=seed, **parameter[n_fold])
            else:
                lgb_params = {
                    "boosting_type": "gbdt",
                    "objective": "binary",
                    # "metric": "auc",
                    "metric": "binary_logloss",
                    "max_depth": 10,
                    "learning_rate": 0.02,
                    "n_estimators": 3072,
                    "colsample_bytree": 1.0,
                    "colsample_bynode": 0.9,
                    "verbose": -1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 10,
                    "extra_trees": True,
                    "num_leaves": 64,
                    "max_bin": 255,
                    "seed": seed,
                }
                model = LGBMClassifier(**lgb_params)

        elif model_name == "catboost":
            if len(parameter) > 0 and len(parameter[n_fold]) > 0:
                model = CatBoostClassifier(**parameter[n_fold])
            else:
                # https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f
                param = {
                    "use_best_model": True,
                    "eval_metric": "CrossEntropy",
                    # "num_boost_round": 3000,
                    # "early_stopping_rounds": 1000,
                    # "depth" : 7,
                    "learning_rate": 0.005,
                    "min_child_samples": 300,  # 0,5,15,300
                    "od_type": "IncToDec",
                    #'subsample': 0.05
                    # "l2_leaf_reg": 2,
                }
                model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "catboost_2":
            # https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f
            param = {
                "use_best_model": True,
                "eval_metric": "CrossEntropy",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                "learning_rate": 0.005,
                "min_child_samples": 30,  # 0,5,15,30
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "catboost_0_2":
            param = {
                "use_best_model": True,
                "eval_metric": "AUC",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                # "learning_rate": 0.005,
                # "min_child_samples": 30,  # 0,5,15,30
                # "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "catboost_3":
            # https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f
            param = {
                "use_best_model": True,
                "eval_metric": "CrossEntropy",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                "learning_rate": 0.005,
                "min_child_samples": 15,  # 0,5,15,30
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "catboost_4":
            # https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f
            param = {
                "use_best_model": True,
                "eval_metric": "CrossEntropy",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                "learning_rate": 0.005,
                "min_child_samples": 5,  # 0,5,15,30
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "lightgbm_binary_2":
            # https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f

            lgb_params = {
                "boosting_type": "dart",
                "objective": "binary",
                "metric": "auc",
                # "metric": "binary_logloss",
                "max_depth": 10,
                "learning_rate": 0.01,
                "n_estimators": 3072,
                "colsample_bytree": 1.0,
                "colsample_bynode": 0.9,
                "verbose": -1,
                "reg_alpha": 0.1,
                "reg_lambda": 10,
                "extra_trees": True,
                "num_leaves": 64,
                "max_bin": 255,
                "seed": seed,
                "num_iterations": 1000,
            }
            model = LGBMClassifier(**lgb_params)

        elif model_name == "catboost_0":
            param = {
                "use_best_model": True,
                "eval_metric": "CrossEntropy",
                # "num_boost_round": 3000,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                # "learning_rate": 0.005,
                # "min_child_samples": 300,  # 0,5,15,300
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)
        elif model_name == "catboost_5":
            param = {
                "use_best_model": True,
                "eval_metric": "AUC",
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "lightgbm":
            model = LGBMRanker(seed=seed, **param)

        elif model_name == "catboost_ranker":
            model = CatBoostRanker(
                random_seed=seed,
                eval_metric="MAP",
                use_best_model=True,
                learning_rate=0.0001,
                # scale_pos_weight=scale_pos_weight,
            )
        elif model_name == "catboost_6":
            param = {
                "use_best_model": True,
                "eval_metric": "AUC",
                "num_boost_round": 3000,
                "scale_pos_weight": 10,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                # "learning_rate": 0.005,
                # "min_child_samples": 300,  # 0,5,15,300
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "catboost_7":
            param = {
                "use_best_model": True,
                "eval_metric": "F1",
                # "num_boost_round": 3000,
                # "scale_pos_weight": 10,
                # "early_stopping_rounds": 1000,
                # "depth" : 7,
                # "learning_rate": 0.005,
                # "min_child_samples": 300,  # 0,5,15,300
                "od_type": "IncToDec",
                #'subsample': 0.05
                # "l2_leaf_reg": 2,
            }
            model = CatBoostClassifier(random_seed=seed, **param)

        elif model_name == "svm":
            model = SVC(probability=True, gamma="auto", random_state=seed)
            model = SkPipeline(model)

        elif model_name == "logistic_regression":
            model = LogisticRegression(random_state=seed)
            model = SkPipeline(model, predict_func_name="predict")
            # model = make_pipeline(StandardScaler(), SVC(gamma="auto"))

        elif model_name == "random_forest":
            model = RandomForestClassifier(random_state=seed)

        elif model_name == "xgboost":
            model = xgb.XGBClassifier(
                booster="gbtree", eval_metric="mlogloss", random_state=seed
            )

        print("===============model===============")
        print(model)
        print("-------------------- " + str(n_fold) + " --------------------")
        # print(train_idx)
        # print(val_idx)
        # train_x, train_y, train_group = (
        #     X.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        #     y.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        #     group.iloc[pd.Index(train_idx).intersection(clean_index) ].reset_index(drop=True),
        # )

        train_x, train_y, train_group = (
            X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            group.iloc[train_idx].reset_index(drop=True),
        )

        test_x, test_y, test_group = (
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True),
            group.iloc[test_idx].reset_index(drop=True),
        )

        gkf_inner = GroupKFold(n_splits=4)
        train_inner_idx, valid_idx = next(
            gkf_inner.split(train_x, train_y, train_group)
        )

        val_x, val_y, valid_group = (
            train_x.iloc[valid_idx].reset_index(drop=True),
            train_y.iloc[valid_idx].reset_index(drop=True),
            train_group.iloc[valid_idx].reset_index(drop=True),
        )
        train_x, train_y, train_group = (
            train_x.iloc[train_inner_idx].reset_index(drop=True),
            train_y.iloc[train_inner_idx].reset_index(drop=True),
            train_group.iloc[train_inner_idx].reset_index(drop=True),
        )
        # print(train_group)
        # print(valid_group)

        print(
            "train_x",
            train_x.shape,
            "train_y",
            train_y.shape,
            "train_group",
            train_group.shape,
        )
        print(
            "val_x", val_x.shape, "val_y", val_y.shape, "valid_group", valid_group.shape
        )

        # train_y = train_y[train_x["scibert_ce_score"].notnull()]
        # train_group = train_group[train_x["scibert_ce_score"].notnull()]
        # train_x = train_x[train_x["scibert_ce_score"].notnull()]

        _train_group = train_group.value_counts().sort_index().values
        _valid_group = valid_group.value_counts().sort_index().values

        test_pred = np.zeros(len(test_idx))

        if under_sampling:
            train_x = run_under_sampling(train_x, seed=seed, neg_scale=5)

        train_x = train_x[cols]
        val_x = val_x[cols]
        test_x = test_x[cols]

        if model_name == "lightgbm":

            model.fit(
                train_x,
                train_y,
                group=_train_group,
                eval_set=[(val_x, val_y)],
                eval_group=[_valid_group],
                eval_metric="map",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=1000, verbose=True),
                    # lgb.log_evaluation(0),
                ],
            )

            # tr_pred = model.predict(train_x, num_iteration=model.best_iteration_)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration_)
            train_pred = model.predict(train_x, num_iteration=model.best_iteration_)
            valid_pred = model.predict(val_x, num_iteration=model.best_iteration_)

            imps = model.feature_importances_
            imps_list.append(imps)

        elif model_name == "lightgbm_binary" or model_name == "lightgbm_binary_2":

            import re

            train_x = train_x.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
            val_x = val_x.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                # eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=1000, verbose=True),
                    # lgb.log_evaluation(0),
                ],
            )
            test_pred = model.predict_proba(
                test_x, num_iteration=model.best_iteration_
            )[:, 1]
            train_pred = model.predict_proba(
                train_x, num_iteration=model.best_iteration_
            )[:, 1]
            valid_pred = model.predict_proba(
                val_x, num_iteration=model.best_iteration_
            )[:, 1]

            imps = model.feature_importances_
            imps_list.append(imps)

        elif (
            model_name == "catboost"
            or model_name == "catboost_0"
            or model_name == "catboost_2"
            or model_name == "catboost_3"
            or model_name == "catboost_4"
            or model_name == "catboost_5"
            or model_name == "catboost_6"
            or model_name == "catboost_7"
        ):
            train_pool = Pool(data=train_x, label=train_y)
            valid_pool = Pool(data=val_x, label=val_y)

            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
            )
            test_pred = model.predict_proba(test_x)[:, 1]
            train_pred = model.predict_proba(train_x)[:, 1]
            valid_pred = model.predict_proba(val_x)[:, 1]
            imps = model.get_feature_importance(train_pool)
            imps_list.append(imps)
            # print(val_pred[:10])

        elif model_name == "catboost_0_2":

            smote = SMOTE(random_state=42)
            train_x_smote, train_y_smote = smote.fit_resample(train_x, train_y)
            train_pool = Pool(data=train_x_smote, label=train_y_smote)
            valid_pool = Pool(data=val_x, label=val_y)

            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
            )
            test_pred = model.predict_proba(test_x)[:, 1]
            train_pred = model.predict_proba(train_x)[:, 1]
            valid_pred = model.predict_proba(val_x)[:, 1]
            imps = model.get_feature_importance(train_pool)
            imps_list.append(imps)

        elif model_name == "catboost_ranker":
            train_pool = Pool(data=train_x, label=train_y, group_id=train_group)
            valid_pool = Pool(data=val_x, label=val_y, group_id=valid_group)

            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
            )
            test_pred = model.predict(test_x)
            train_pred = model.predict(train_x)
            valid_pred = model.predict(val_x)
            imps = model.get_feature_importance(train_pool)
            imps_list.append(imps)

        elif model_name == "svm":
            model.fit(train_x, train_y, val_x, val_y)
            test_pred = model.predict(test_x)
            train_pred = model.predict(train_x)
            valid_pred = model.predict(val_x)
            imps = [0 for _ in range(len(train_x.columns))]
            # imps = model.coef_
            imps_list.append(imps)

        elif model_name == "logistic_regression":
            model.fit(train_x, train_y, val_x, val_y)
            test_pred = model.predict(test_x)
            train_pred = model.predict(train_x)
            valid_pred = model.predict(val_x)
            imps = model.coef_[0]
            print(imps)
            imps_list.append(imps)

        elif model_name == "random_forest":
            model.fit(train_x, train_y)
            test_pred = model.predict_proba(test_x)[:, 1]
            train_pred = model.predict_proba(train_x)[:, 1]
            valid_pred = model.predict_proba(val_x)[:, 1]
            imps = model.feature_importances_
            imps_list.append(imps)

        elif model_name == "xgboost":
            model.fit(train_x, train_y)
            test_pred = model.predict_proba(test_x)[:, 1]
            train_pred = model.predict_proba(train_x)[:, 1]
            valid_pred = model.predict_proba(val_x)[:, 1]
            imps = model.feature_importances_
            imps_list.append(imps)

        # train_preds[train_idx] = tr_pred
        oof_preds[test_idx] = test_pred

        # score
        mapk = evaluate_mapk(test_y, test_pred, pd.DataFrame({"pid": test_group}))
        map_score.append(mapk)

        train_mapk = evaluate_mapk(
            train_y, train_pred, pd.DataFrame({"pid": train_group})
        )
        train_map_score.append(train_mapk)

        valid_mapk = evaluate_mapk(
            val_y, valid_pred, pd.DataFrame({"pid": valid_group})
        )
        valid_map_score.append(valid_mapk)

        auc = roc_auc_score(test_y, test_pred)
        auc_score.append(auc)

        train_auc = roc_auc_score(train_y, train_pred)
        train_auc_score.append(train_auc)

        valid_auc = roc_auc_score(val_y, valid_pred)
        valid_auc_score.append(valid_auc)

        print(
            "mapk",
            mapk,
            "auc",
            auc,
            "train_mapk",
            train_mapk,
            "train_auc",
            train_auc,
            "valid_mapk",
            valid_mapk,
            "valid_auc",
            valid_auc,
        )

        models.append(model)

    val_score = {
        "map": np.mean(map_score),
        "auc": np.mean(auc_score),
        "train_map": np.mean(train_map_score),
        "train_auc": np.mean(train_auc_score),
        "valid_map": np.mean(valid_map_score),
        "valid_auc": np.mean(valid_auc_score),
        "map_list": map_score,
        "auc_list": auc_score,
        "train_map_list": train_map_score,
        "train_auc_list": train_auc_score,
        "valid_map_list": valid_map_score,
        "valid_auc_list": valid_auc_score,
    }

    # FI
    imps = np.mean(imps_list, axis=0)
    df_fi = pd.DataFrame({"fea_name": train_x.columns, "fi_score": imps})
    df_fi = df_fi.sort_values("fi_score", ascending=False)

    return oof_preds, train_preds, df_fi, val_score, models


def prediction_batch(
    eval_features: pd.DataFrame,
    models: List[lgb.LGBMClassifier],
    feature_columns: List[str],
    model_name: str,
    return_df: bool = False,
):
    DATA_TRACE_DIR = "../dataset"
    data_dir = join(DATA_TRACE_DIR, "PST")
    papers = load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    sub_dict = {}
    pid_list = []
    bid_list = []
    score_list = []

    score_list_dict = {f"model_{i}": [] for i in range(len(models))}

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        cur_feature = eval_features[eval_features["pid"] == cur_pid]
        cur_feature.loc[:, ("bid_int",)] = cur_feature["bid"].apply(
            lambda x: int(x[1:])
        )
        cur_feature = cur_feature.sort_values("bid_int").reset_index(drop=True)

        x = cur_feature[feature_columns]

        preds = []
        # print(cur_ref_feature)
        pid_list.extend([cur_pid] * cur_feature.shape[0])
        bid_list.extend(cur_feature["bid"].tolist())

        for i, model in enumerate(models):
            if model_name == "lightgbm":
                pred = model.predict(x, num_iteration=model.best_iteration_)[0]
                # preds.append(pred)
            elif model_name == "lightgbm_binary" or model_name == "lightgbm_binary_2":
                pred = model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1]
                # preds.append(pred)
            elif (
                model_name == "catboost"
                or model_name == "catboost_0"
                or model_name == "catboost_2"
                or model_name == "catboost_3"
                or model_name == "catboost_4"
                or model_name == "catboost_5"
                or model_name == "catboost_6"
                or model_name == "catboost_7"
                or model_name == "catboost_0_2"
            ):
                pred = model.predict_proba(x)[:, 1]
                # preds.append(pred)
            elif model_name == "catboost_ranker":
                pred = model.predict(x)
            elif model_name == "svm":
                pred = model.predict(x)
            elif model_name == "logistic_regression":
                pred = model.predict(x)
            elif model_name == "random_forest":
                pred = model.predict_proba(x)[:, 1]

            # preds.append(pred)
            score_list_dict[f"model_{i}"].extend(pred)

        # sub_dict[cur_pid] = np.mean(preds, axis=0).tolist()
        # score_list.extend(preds)
        # score_list.extend(sub_dict[cur_pid])

    # print(score_list)
    score_list_dict["pid"] = pid_list
    score_list_dict["bid"] = bid_list

    df = pd.DataFrame({"pid": pid_list, "bid": bid_list})
    df_score = pd.DataFrame(score_list_dict)

    df = pd.concat([df, df_score], axis=1)
    df["mean"] = df[[f"model_{i}" for i in range(len(models))]].mean(axis=1)

    return df


import argparse


class ParamProcessor(argparse.Action):
    """
    https://qiita.com/Hi-king/items/de960b6878d6280eaffc
    --param foo=a型の引数を辞書に入れるargparse.Action
    """

    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace, self.dest, [])
        if param_dict is None:
            param_dict = {}

        k, v = values.split("=")
        param_dict[k] = v
        setattr(namespace, self.dest, param_dict)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lightgbm")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--scale_pos_weight", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param", action=ParamProcessor, default={})
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../classifier_output")
    parser.add_argument("--trial_name", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--under_sampling", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    df_train_feature = pd.read_csv(
        f"{args.input_dir}/df_train_feature.csv", index_col=False
    )
    df_test_feature = pd.read_csv(
        f"{args.input_dir}/df_test_feature.csv", index_col=False
    )

    if args.fill:
        df_train_feature = fill_missing_group_mean(df_train_feature)
        df_test_feature = fill_missing_group_mean(df_test_feature)

    X = df_train_feature.drop(["pid", "bid", "ref_pid", "label"], axis=1)
    y = df_train_feature["label"]
    cols = X.columns
    if len(args.param) == 0:
        best_params = []
    else:
        best_params = [args.param for _ in range(args.k)]

    pid_unique = df_train_feature["pid"].unique()
    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(pid_unique)

    group_map = {pid: i for i, pid in enumerate(pid_unique)}
    group = df_train_feature["pid"].map(group_map)

    oof_preds, train_preds, df_fi, val_score, models = train_k_fold(
        df_train_feature,
        y,
        cols,
        group,
        None,
        n_splits=args.k,
        model_name=args.model_name,
        scale_pos_weight=args.scale_pos_weight,
        parameter=best_params,
        seed=args.seed,
        under_sampling=args.under_sampling,
    )

    print(df_fi.head(30))
    print(val_score)

    df_oof = df_train_feature[["pid", "bid", "label"]].copy()
    df_oof["oof"] = oof_preds

    os.makedirs(f"{args.output_dir}/{args.trial_name}", exist_ok=True)

    df_oof.to_csv(
        f"{args.output_dir}/{args.trial_name}/oof_prediction.csv", index=False
    )

    with open(f"{args.output_dir}/{args.trial_name}/params.yaml", "w") as f:
        yaml.dump(vars(args), f)

    pred = prediction_batch(
        df_test_feature, models, cols, args.model_name, return_df=True
    )
    pred.to_csv(
        f"{args.output_dir}/{args.trial_name}/test_pub_prediction.csv", index=False
    )
    df_fi.to_csv(
        f"{args.output_dir}/{args.trial_name}/feature_importance.csv", index=False
    )
    dump_json(val_score, f"{args.output_dir}/{args.trial_name}", "score.json")

    # submit = {}
    # for pid, _df in pred.groupby("pid"):
    #     _df["bid_int"] = _df["bid"].apply(lambda x: int(x[1:]))
    #     _df = _df.sort_values("bid_int")
    #     submit[pid] = _df["score"].tolist()

    # json.dump(submit, open(f"{args.output_dir}/{args.trial_name}/submit.json", "w"))

    # for i, model in enumerate(models):
    #     pickle.dump(
    #         model,
    #         open(
    #             f"{args.output_dir}/{args.trial_name}/model_{i}_{args.model_name}.pkl",
    #             "wb",
    #         ),
    #     )


if __name__ == "__main__":
    main()
