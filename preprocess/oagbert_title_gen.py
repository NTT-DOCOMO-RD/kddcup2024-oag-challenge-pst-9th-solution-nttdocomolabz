import pandas as pd
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


def run(df, model):
    gen_ref_title = []
    pid_list = []
    # bid_list = []

    venue_col = "venue"
    authors_col = "authors"
    affiliations_col = "org"
    concepts_col = "keywords"
    abstract_col = "abstract"
    introduction_col = "introduction"

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row["title"] is not None and type(row["title"]) != float:
            continue

        # context = (
        #     row["context_clean"]
        #     if (row["context_clean"] is not None)
        #     and type(row["context_clean"]) != float
        #     else ""
        # )
        # context = context.replace("\n", " ")
        # context = context.replace("\t", " ")

        venue = (
            row[venue_col]
            if (row[venue_col] is not None) and type(row[venue_col]) != float
            else ""
        )
        authors = (
            row[authors_col]
            if row[authors_col] is not None and type(row[authors_col]) != float
            else []
        )
        affiliations = (
            row[affiliations_col]
            if row[affiliations_col] is not None
            and type(row[affiliations_col]) != float
            else []
        )
        concepts = (
            row[concepts_col]
            if row[concepts_col] is not None and type(row[concepts_col]) != float
            else []
        )
        abstract = (
            row[abstract_col]
            if row[abstract_col] is not None and type(row[abstract_col]) != float
            else ""
        )

        introduction = (
            row[introduction_col]
            if row[introduction_col] is not None
            and type(row[introduction_col]) != float
            else ""
        )
        text = abstract if abstract != "" else introduction

        try:
            gen_titles = model.generate_title(
                abstract=text,
                authors=authors,
                venue=venue,
                affiliations=affiliations,
                concepts=concepts,
            )
            gen_title = gen_titles[0][0]

        except Exception as e:
            print(e)

            print("pid", row["pid"])
            print("bid", row["bid"])
            print("abstract", abstract)
            print("authors", authors)
            print("venue", venue)
            print("affiliations", affiliations)
            print("concepts", concepts)
            continue

        print(gen_title)
        gen_ref_title.append(gen_title)
        pid_list.append(row["pid"])
        # bid_list.append(row["bid"])

    df = pd.DataFrame(
        {
            "pid": pid_list,
            "gen_title": gen_ref_title,
        }
    )
    return df


def main():
    base_dir = "../data"
    # df_train_context = pd.read_csv(f"{base_dir}/train_context.csv")
    # df_test_context = pd.read_csv(f"{base_dir}/test_context.csv")
    df_test_pub_context = pd.read_csv(f"{base_dir}/test_pub_context.csv")
    _df = df_test_pub_context.drop_duplicates(["pid"]).reset_index(drop=True)

    print(_df)

    # df_train_context = df_train_context.head(30)
    # df_test_context = df_test_context.head(30)

    # tokenizer, model = oagbert("oagbert-v2-sim")
    tokenizer, model = oagbert("oagbert-v2-lm")
    model.eval()

    # test pub
    df_gen_title = run(_df, model)
    df_gen_title.to_csv(f"{base_dir}/test_pub_gen_title.csv", index=False)
    print(df_gen_title)
    df_test_pub_context = pd.merge(
        df_test_pub_context, df_gen_title, on="pid", how="left"
    )
    df_test_pub_context["title"] = df_test_pub_context["title"].fillna(
        df_test_pub_context["gen_title"]
    )
    df_test_pub_context = df_test_pub_context.drop(columns=["gen_title"])
    df_test_pub_context.to_csv(
        f"{base_dir}/test_pub_context_gen_title.csv", index=False
    )

    # train
    # df_gen_ref_title = run(df_train_context, model)
    # df_gen_ref_title.to_csv(f"{base_dir}/train_gen_ref_title.csv", index=False)

    # df_train_context = pd.merge(
    #     df_train_context, df_gen_ref_title, on=["pid", "bid"], how="left"
    # )
    # df_train_context["ref_title"] = df_train_context["ref_title"].fillna(
    #     df_train_context["gen_ref_title"]
    # )
    # df_train_context = df_train_context.drop(columns=["gen_ref_title"])

    # # test
    # df_gen_ref_title = run(df_test_context, model)
    # df_gen_ref_title.to_csv(f"{base_dir}/test_gen_ref_title.csv", index=False)
    # df_test_context = pd.merge(
    #     df_test_context, df_gen_ref_title, on=["pid", "bid"], how="left"
    # )
    # df_test_context["ref_title"] = df_test_context["ref_title"].fillna(
    #     df_test_context["gen_ref_title"]
    # )
    # df_test_context = df_test_context.drop(columns=["gen_ref_title"])

    # df_train_context.to_csv(f"{base_dir}/train_context_gen_ref_title.csv", index=False)
    # df_test_context.to_csv(f"{base_dir}/test_context_gen_ref_title.csv", index=False)


if __name__ == "__main__":
    main()
