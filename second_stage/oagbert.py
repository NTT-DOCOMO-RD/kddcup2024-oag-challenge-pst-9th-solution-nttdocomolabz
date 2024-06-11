from cogdl.oag import oagbert
from tqdm import tqdm
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import unicodedata
import string
import torch
from sklearn.decomposition import TruncatedSVD
import numpy as np
import math

import nltk

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


def encode_oag_bert(model, df, mode="default"):

    assert mode in ["default", "ref"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embs = []
    model.bert.to(device)
    for i, row in tqdm(df.iterrows(), total=len(df)):

        if mode == "default":
            title_col = "title"
            abstract_col = "abstract"
            authors_col = "authors"
            venue_col = "venue"
            org_col = "org"
            keywords_col = "keywords"
        elif mode == "ref":
            title_col = "ref_title"
            abstract_col = "ref_abstract"
            authors_col = "ref_authors"
            venue_col = "ref_venue"
            org_col = "ref_org"
            keywords_col = "ref_keywords"

        # nan の場合floatなので、strに変換
        title = (
            row[title_col]
            if (row[title_col] is not None and type(row[title_col]) != float)
            else ""
        )

        if (
            mode == "ref"
            and row["context_clean"] is not None
            and type(row["context_clean"]) != float
        ):
            abstract = (
                row[abstract_col]
                if (row[abstract_col] is not None and type(row[abstract_col]) != float)
                else row["context_clean"]
            )

        else:
            abstract = (
                row[abstract_col]
                if (row[abstract_col] is not None and type(row[abstract_col]) != float)
                else ""
            )
        authors = (
            eval(row[authors_col])
            if (row[authors_col] is not None and type(row[authors_col]) != float)
            else []
        )

        venue = (
            row[venue_col]
            if (row[venue_col] is not None and type(row[venue_col]) != float)
            else ""
        )

        affiliations = (
            eval(row[org_col])
            if (row[org_col] is not None and type(row[org_col]) != float)
            else []
        )
        concepts = (
            eval(row[keywords_col])
            if (row[keywords_col] is not None and type(row[keywords_col]) != float)
            else []
        )

        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = model.build_inputs(
            title=title,
            abstract=abstract,
            venue=venue,
            authors=authors,
            concepts=concepts,
            affiliations=affiliations,
        )

        try:
            _, embed = model.bert.forward(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
                token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
                attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
                position_ids_second=torch.LongTensor(position_ids_second).to(device),
                # mask_propmt_text=(row["context"] if mode == "ref" else ""),
            )
            x = embed.detach().cpu().numpy()

        except Exception as e:
            print(e)
            print("pid", row["pid"])
            print("ref_pid", row["ref_pid"])
            print(
                "title",
                title,
            )
            print("abstract", abstract)
            print("venue", venue)
            print("authors", authors)
            print("concepts", concepts)
            print("affiliations", affiliations)
            # break
            x = np.zeros((1, 768))

        embs.append(x)

    embs = np.concatenate(embs, axis=0)

    return embs


def encode_oag_bert_title(model, df, mode="default"):

    assert mode in ["default", "ref"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embs = []
    model.bert.to(device)
    for i, row in tqdm(df.iterrows(), total=len(df)):

        if mode == "default":
            title_col = "title"
            abstract_col = "abstract"
            authors_col = "authors"
            venue_col = "venue"
            org_col = "org"
            keywords_col = "keywords"
        elif mode == "ref":
            title_col = "ref_title"
            abstract_col = "ref_abstract"
            authors_col = "ref_authors"
            venue_col = "ref_venue"
            org_col = "ref_org"
            keywords_col = "ref_keywords"
            context = "context_clean"

        # nan の場合floatなので、strに変換
        title = (
            row[title_col]
            if (row[title_col] is not None and type(row[title_col]) != float)
            else ""
        )

        if mode == "ref":
            context = (
                row[context]
                if (row[context] is not None and type(row[context]) != float)
                else ""
            )

            (
                input_ids,
                input_masks,
                token_type_ids,
                masked_lm_labels,
                position_ids,
                position_ids_second,
                masked_positions,
                num_spans,
            ) = model.build_inputs(
                title=title,
                # abstract=abstract,
                # venue=venue,
                # authors=authors,
                # concepts=concepts,
                # eaffiliations=affiliations,
                mask_propmt_text=f"This paper is referenced in the next sentence.{context}",
            )
        else:
            (
                input_ids,
                input_masks,
                token_type_ids,
                masked_lm_labels,
                position_ids,
                position_ids_second,
                masked_positions,
                num_spans,
            ) = model.build_inputs(title=title)

        try:
            _, embed = model.bert.forward(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
                token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
                attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
                position_ids_second=torch.LongTensor(position_ids_second).to(device),
                # mask_propmt_text=(row["context"] if mode == "ref" else ""),
            )
            x = embed.detach().cpu().numpy()

        except Exception as e:
            print(e)
            print("pid", row["pid"])
            print("ref_pid", row["ref_pid"])
            print(
                "title",
                title,
            )
            # break
            x = np.zeros((1, 768))

        embs.append(x)

    embs = np.concatenate(embs, axis=0)

    return embs


def decompose(
    df,
    embs,
    model=None,
    n_dim=20,
    seed=42,
    col_prefix="oag_bert",
):
    if model is None:
        model = TruncatedSVD(n_components=n_dim, random_state=seed).fit(embs)
        vector = model.transform(embs)
    else:
        vector = model.transform(embs)

    df_vector = pd.DataFrame(
        vector, columns=[f"{col_prefix}_{i}" for i in range(n_dim)]
    )
    df_vector = pd.concat([df[["pid", "bid"]], df_vector], axis=1)

    return df_vector, model


def calc_cos_sim(vector1, vector2):

    norm1 = np.linalg.norm(vector1, axis=1)
    norm2 = np.linalg.norm(vector2, axis=1)
    if np.all(norm1 == 0) or np.all(norm2 == 0):
        return np.zeros(len(vector1))

    return (vector1 * vector2).sum(axis=1) / (
        np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1)
    )


def similarity(df, emb1, emb2):
    sim = calc_cos_sim(emb1, emb2)
    df["oag_bert_cos_sim"] = sim
    return df[["pid", "bid", "oag_bert_cos_sim"]]


def run(df_context, model, base_dir, svd=None, ref_svd=None):

    embs = encode_oag_bert(model, df_context)
    train_index = df_context[df_context["label"] != -2].index
    test_index = df_context[df_context["label"] == -2].index

    np.save(f"{base_dir}/train_context_oagbert.npy", embs[train_index])
    np.save(f"{base_dir}/test_pub_context_oagbert.npy", embs[test_index])

    ref_embs = encode_oag_bert(model, df_context, mode="ref")
    np.save(f"{base_dir}/train_context_oagbert_ref.npy", ref_embs[train_index])
    np.save(f"{base_dir}/test_pub_context_oagbert_ref.npy", ref_embs[test_index])

    # train_embs = np.load(f"{base_dir}/train_context_oagbert.npy")
    # train_ref_embs = np.load(f"{base_dir}/test_context_oagbert_ref.npy")

    # test_embs = np.load(f"{base_dir}/test_pub_context_oagbert.npy")
    # test_ref_embs = np.load(f"{base_dir}/test_pub_context_oagbert_ref.npy")

    df_sim = similarity(df_context, embs, ref_embs)
    print(df_sim)

    df_sim.loc[train_index].to_csv(f"{base_dir}/train_oagbert_sim.csv", index=False)
    df_sim.loc[test_index].to_csv(f"{base_dir}/test_pub_oagbert_sim.csv", index=False)

    df_vector, svd = decompose(df_context, embs, n_dim=20, model=svd)
    df_vector.loc[train_index].to_csv(
        f"{base_dir}/train_oagbert_vector.csv", index=False
    )
    df_vector.loc[test_index].to_csv(
        f"{base_dir}/test_pub_oagbert_vector.csv", index=False
    )


def run_title(df_context, model, base_dir, svd=None, ref_svd=None):

    embs = encode_oag_bert_title(model, df_context)
    train_index = df_context[df_context["label"] != -2].index
    test_index = df_context[df_context["label"] == -2].index

    np.save(f"{base_dir}/train_title_oagbert.npy", embs[train_index])
    np.save(f"{base_dir}/test_pub_title_oagbert.npy", embs[test_index])

    ref_embs = encode_oag_bert(model, df_context, mode="ref")
    np.save(f"{base_dir}/train_title_oagbert_ref.npy", ref_embs[train_index])
    np.save(f"{base_dir}/test_pub_title_oagbert_ref.npy", ref_embs[test_index])

    # train_embs = np.load(f"{base_dir}/train_context_oagbert.npy")
    # train_ref_embs = np.load(f"{base_dir}/test_context_oagbert_ref.npy")

    # test_embs = np.load(f"{base_dir}/test_pub_context_oagbert.npy")
    # test_ref_embs = np.load(f"{base_dir}/test_pub_context_oagbert_ref.npy")

    df_sim = similarity(df_context, embs, ref_embs)
    print(df_sim)

    df_sim.loc[train_index].to_csv(
        f"{base_dir}/train_oagbert_title_sim.csv", index=False
    )
    df_sim.loc[test_index].to_csv(
        f"{base_dir}/test_pub_oagbert_title_sim.csv", index=False
    )

    df_vector, svd = decompose(df_context, embs, n_dim=20, model=svd)
    df_vector.loc[train_index].to_csv(
        f"{base_dir}/train_oagbert_title_vector.csv", index=False
    )
    df_vector.loc[test_index].to_csv(
        f"{base_dir}/test_pub_oagbert_title_vector.csv", index=False
    )


def main():
    base_dir = "../data"
    df_train_context = pd.read_csv(f"{base_dir}/train_context.csv")
    df_test_pub_context = pd.read_csv(
        f"{base_dir}/test_pub_gen_context_filled_citation.csv"
    )
    df_test_pub_context = df_test_pub_context.drop(
        ["n_citation", "ref_n_citation"], axis=1
    )
    df_test_pub_context["label"] = -2

    print(df_train_context.shape, df_test_pub_context.shape)
    df_context = pd.concat([df_train_context, df_test_pub_context]).reset_index(
        drop=True
    )

    # df_context = pd.concat(
    #    [df_train_context.head(30), df_test_pub_context.head(30)]
    # ).reset_index(drop=True)

    print(df_context.shape)

    # tokenizer, model = oagbert("oagbert-v2-sim")
    tokenizer, model = oagbert("oagbert-v2")
    model.eval()

    # train
    # run(df_context, model, base_dir)
    run_title(df_context, model, base_dir)
    # test
    # run(df_test_context, model, "test", base_dir, svd, ref_svd)


if __name__ == "__main__":
    main()
