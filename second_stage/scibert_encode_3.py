import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import unicodedata
import nltk
import string
import torch
from sklearn.decomposition import TruncatedSVD
import numpy as np
from fuzzywuzzy import fuzz
import torch

nltk.download("stopwords")


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


def text_to_vector(text_list: List[str], model_name: str = "scibert"):
    text_list = [str_normalize(text) for text in text_list]

    if model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
        model = SentenceTransformer(BERT_MODEL)
    else:
        raise ValueError(f"unkown word")

    vector = model.encode(text_list, show_progress_bar=True)

    return vector


def calc_inner_product(vector1, vector2):
    return (vector1 * vector2).sum(axis=1)


def calc_cos_sim(vector1, vector2):
    return (vector1 * vector2).sum(axis=1) / (
        np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1)
    )


def calc_cos_sim_torch(vector1, vector2):
    vector1 = torch.from_numpy(vector1).to("cuda")
    vector2 = torch.from_numpy(vector2).to("cuda")
    return torch.nn.functional.cosine_similarity(vector1, vector2).to("cpu").numpy()


def common_word_count(text1, text2):
    text1_list = text1.lower().split()
    text2_list = text2.lower().split()
    count = 0
    for word in text1_list:
        if word in text2_list:
            count += 1
    return count


def common_word_count_normalize(text1, text2):
    text1_list = text1.lower().split()
    text2_list = text2.lower().split()
    count = 0
    for word in text1_list:
        if word in text2_list:
            count += 1

    # 0除算を避ける
    if len(text1_list) + len(text2_list) == 0:
        return 0

    count = count / (len(text1_list) + len(text2_list))
    return count


def encode(df, text_series, model_name, col_prefix, n_dim=20, model=None, seed=42):
    text_series = text_series.fillna("").apply(lambda x: x.lower()).tolist()
    vector_orgn = text_to_vector(text_series, model_name=model_name)
    if model is None:
        model = TruncatedSVD(n_components=n_dim, random_state=seed).fit(vector_orgn)
        vector = model.transform(vector_orgn)
    else:
        vector = model.transform(vector_orgn)

    df_vector = pd.DataFrame(
        vector, columns=[f"{col_prefix}_{i}" for i in range(n_dim)]
    )
    df_vector = pd.concat([df[["pid", "bid", "label"]], df_vector], axis=1)
    return df_vector, vector_orgn, model


def run_encode(df, model_name, svd_models={}, n_dim=20, out_dir="../script/data/"):

    # if mode == "train":
    #    assert len(svd_models) == 0

    # elif mode == "test":
    #    assert len(svd_models) > 0

    #####################
    # prepare data concat
    df["title_abstract"] = df["title"] + " " + df["abstract"]
    df["ref_title_ref_abstract"] = df["ref_title"] + " " + df["ref_abstract"]

    df["title_abstract_keyword"] = df["title_abstract"] + " " + df["keywords"]
    df["ref_title_ref_abstract_ref_keyword"] = (
        df["ref_title_ref_abstract"] + " " + df["ref_keywords"]
    )

    df["title_abstract_context_keyword"] = (
        df["title_abstract"] + " " + df["context"] + " " + df["keywords"]
    )

    ##############################
    # encode
    # encode target column
    columns = [
        "title",
        "abstract",
        "context",
        "keywords",
        "venue",
        "org",
        "introduction",
        "conclusion",
        "related_work",
        "title_abstract",
        "title_abstract_keyword",
        "title_abstract_context_keyword",
    ]
    ref_columns = [
        "context",
        "ref_title",
        "ref_abstract",
        "ref_keywords",
        "ref_venue",
        "ref_org",
        "ref_introduction",
        # "ref_conclusion",
        # "ref_related_work",
        "ref_title_ref_abstract",
        "ref_title_ref_abstract_ref_keyword",
    ]

    vectors = {}
    ref_vectors = {}
    other_vectors = {}

    svd = None
    for col in columns + ref_columns:
        print(f"Encode {col}")
        if col in vectors or col in ref_vectors:
            continue

        df_vector, vector, svd = encode(
            df,
            df[col],
            model_name,
            f"{model_name}_{col}_truncate_svd",
            n_dim=n_dim,
            model=svd,
        )

        df_vector_train = df_vector[
            (df_vector["label"] != -2) & (df_vector["label"] != -1)
        ].reset_index(drop=True)
        df_vector_test_pub = df_vector[df_vector["label"] == -2].reset_index(drop=True)

        df_vector_train.drop("label", axis=1).to_csv(
            f"{out_dir}/train_{col}_vector.csv", index=False
        )
        df_vector_test_pub.drop("label", axis=1).to_csv(
            f"{out_dir}/test_pub_{col}_vector.csv", index=False
        )

        if col in columns:
            vectors[col] = vector

        if col in ref_columns:
            ref_vectors[col] = vector

        svd_models[col] = svd

    # other
    _, vector_sentence_inspired, _ = encode(
        df,
        pd.Series(["main idea of this paper is inspired by the reference"] * len(df)),
        model_name,
        f"{model_name}_sentence_inspired_truncate_svd",
        n_dim=n_dim,
        model=None,
    )
    other_vectors["sentence_inspired"] = vector_sentence_inspired

    _, vector_sentence_core, _ = encode(
        df,
        pd.Series(
            ["the core method of this paper is derived from the reference"] * len(df)
        ),
        model_name,
        f"{model_name}_sentence_core_truncate_svd",
        n_dim=n_dim,
        model=None,
    )
    other_vectors["sentence_core"] = vector_sentence_core

    _, vector_sentence_essential, _ = encode(
        df,
        pd.Series(
            [
                "the reference is essential for this paper without the work of this reference, this paper cannot be completed"
            ]
            * len(df)
        ),
        model_name,
        f"{model_name}_sentence_essential_truncate_svd",
        n_dim=n_dim,
        model=None,
    )

    other_vectors["sentence_essential"] = vector_sentence_essential

    _, vector_sentence_all, _ = encode(
        df,
        pd.Series(
            [
                " ".join(
                    [
                        "main idea of this paper is inspired by the reference",
                        "the core method of this paper is derived from the reference",
                        "the reference is essential for this paper without the work of this reference, this paper cannot be completed",
                    ]
                )
            ]
            * len(df)
        ),
        model_name,
        f"{model_name}_sentence_all_truncate_svd",
        n_dim=n_dim,
        model=None,
    )
    other_vectors["sentence_all"] = vector_sentence_all

    #################################################
    # similarity feature

    sim_data = {
        "pid": df["pid"].copy(),
        "bid": df["bid"].copy(),
        "label": df["label"].copy(),
    }

    for key, vec in vectors.items():
        for ref_key, ref_vec in ref_vectors.items():
            if key == ref_key:
                continue

            sim_data[f"{model_name}_cos_sim_{key}_{ref_key}"] = calc_cos_sim_torch(
                vec, ref_vec
            )
            # sim_data[f"{model_name}_inner_product_{key}_{ref_key}"] = (
            #    calc_inner_product(vec, ref_vec)
            # )

    vector_c = vectors["context"]
    for key, vec in other_vectors.items():
        sim_data[f"{model_name}_cos_sim_c_{key}"] = calc_cos_sim_torch(vector_c, vec)
        # sim_data[f"{model_name}_inner_product_c_{key}"] = calc_inner_product(
        #    vector_c, vec
        # )

    df_sim = pd.DataFrame(sim_data)

    #####################################
    # wourd count feature
    dfs = []
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(ref_columns):

            if col_i == col_j:
                continue

            text1 = df[col_i].fillna("").copy().tolist()
            text2 = df[col_j].fillna("").copy().tolist()
            common_word_count_list = [
                common_word_count(t1, t2) for t1, t2 in zip(text1, text2)
            ]

            common_word_count_normalize_list = [
                common_word_count_normalize(t1, t2) for t1, t2 in zip(text1, text2)
            ]

            fuzz_score = [
                fuzz.ratio(t1.lower(), t2.lower()) for t1, t2 in zip(text1, text2)
            ]

            _df = pd.DataFrame(
                {
                    f"common_word_count_{col_i}_{col_j}": common_word_count_list,
                    f"common_word_count_normalize_{col_i}_{col_j}": common_word_count_normalize_list,
                    f"common_word_fuzz_{col_i}_{col_j}": fuzz_score,
                }
            )
            dfs.append(_df)

    df_sim = pd.concat([df_sim] + dfs, axis=1)
    df_sim_train = df_sim[
        (df_sim["label"] != -2) & (df_sim["label"] != -1)
    ].reset_index(drop=True)
    df_sim_test_pub = df_sim[df_sim["label"] == -2].reset_index(drop=True)

    df_sim_train.drop("label", axis=1).to_csv(
        f"{out_dir}/train_{model_name}_sim.csv", index=False
    )
    df_sim_test_pub.drop("label", axis=1).to_csv(
        f"{out_dir}/test_pub_{model_name}_sim.csv", index=False
    )

    return svd_models


def main():
    train_context = "../data/train_context.csv"
    # test_context = "../script/data_0529/test_context.csv"
    # test_pub_context = "data/test_pub_context_gen_title.csv"
    test_pub_context = "../data/test_pub_gen_context_filled_citation.csv"
    output_dir = "../data/"

    df = pd.read_csv(train_context)
    # df_test = pd.read_csv(test_context)
    df_test_pub = pd.read_csv(test_pub_context).drop(
        ["n_citation", "ref_n_citation"], axis=1
    )

    df_test_pub["label"] = -2

    df_all = pd.concat([df, df_test_pub]).reset_index(drop=True)
    svd_models = run_encode(df_all, "scibert", n_dim=20, out_dir=output_dir)

    # run_encode(
    #     df_test,
    #     "scibert",
    #     # svd_models=svd_models,
    #     n_dim=20,
    #     mode="test",
    #     out_dir=output_dir,
    # )


if __name__ == "__main__":
    main()
