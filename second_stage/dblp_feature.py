import pandas as pd
import json
import json
import os
from os.path import join
import csv
import random
from lxml import etree
from fuzzywuzzy import fuzz
import re
from collections import defaultdict as dd
from tqdm import tqdm

# import utils
# import settings

# import logging

import os
from os.path import abspath, dirname, join

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import LGBMRanker
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from typing import List, Dict
from bs4 import BeautifulSoup

# PROJ_DIR = join(abspath(dirname(__file__)))
# DATA_DIR = join(PROJ_DIR, "../dataset/")
# OUT_DIR = join(PROJ_DIR, "out")

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)
DATA_TRACE_DIR = "../dataset"

import pickle


def load_json(rfdir, rfname):
    print("loading %s ...", rfname)
    with open(join(rfdir, rfname), "r", encoding="utf-8") as rf:
        data = json.load(rf)
        print("%s loaded", rfname)
        return data


class Process_data(object):
    def __init__(self, paper_dic, mode, paper_info_more, seed=42):
        self.paper_dic = paper_dic
        # self.paper_number = -1
        paper_id = paper_dic["_id"]
        if mode == "train":
            paper_positive_id = [item["_id"] for item in paper_dic["refs_trace"]]
        self.authors_set = set(
            [item.get("name") for item in paper_dic.get("authors", "")]
        )
        # 通过xml获取tree和listBibl
        try:
            path = f"../dataset/PST/paper-xml/{paper_id}.xml"
            self.tree = etree.parse(path)
            root = self.tree.getroot()
            listBibl = root.xpath("//*[local-name()='listBibl']")[0]
            self.biblStruct = listBibl.getchildren()
            self.num_ref = len(self.biblStruct)
        except OSError:
            self.tree = None
            self.num_ref = 0
            print("not exits xml " + paper_id)
        # 获取论文引用数
        self.reference_num = self.get_reciprocal_of_reference_num()
        if (
            mode == "test"
        ):  # train和valid需要随机选择部分论文构造正例，而test则可以直接把所有例子都放入query_list
            query_list = paper_dic.get("references", [])
        else:

            # references にはnegativeのみにする
            references = paper_dic.get("references", [])
            for item in paper_positive_id:
                try:
                    references.remove(item)
                except ValueError:
                    continue

            random.seed(seed)

            # 正例・負例はそのまま
            query_list = references

            # 正例・負例の数を合わせる
            # query_list = random.sample(
            #    references, min(max(len(paper_positive_id), 1), len(references))
            # )  # 最少选一个负例,最多references个（存在正例数多于reference的情况！）
            query_list += paper_positive_id

        self.data_list = []
        self.label_list = []
        self.ref_pid_list = []

        # 引用文献ごとに特徴量を取得する
        for i, item in enumerate(query_list):
            this_data = []
            self.query_result = paper_info_more.get(item, {})
            reference_place_list = self.get_referenced_place_num(item)
            if len(reference_place_list) == 0:
                self.data_list.append([])
                continue  # 如果返回长度为0说明文章标题在xml的reference中没有找到自己的序号，这里暂时先不管它。
            this_data.append(self.get_referenced_num())
            this_data.append(self.get_common_authors(item))
            this_data.append(self.reference_num)
            this_data.append(self.key_words())
            this_data += reference_place_list
            self.data_list.append(this_data)
            self.ref_pid_list.append(item)
            if mode == "train":
                self.label_list.append([1] if item in paper_positive_id else [0])

        self.columns = [
            "referenced_num",
            "has_common_authors",
            "reciprocal_of_reference_num",
            "has_common_keyword",
            "ref_abstract",
            "ref_introduction",
            "ref_related_work",
            "ref_method",
            "ref_graph_and_figure",
            "ref_result",
            "ref_others",
            "ref_count",
        ]

    # ONE 被引用次数
    def get_referenced_num(self):
        return self.query_result.get("n_citation", 0)

    # TWO,SIX,EIGHT 引用位置, 是否出现在图表中, 引用次数/引用总数
    # 論文引用がどこで行われているかを抽出する。
    # 0 abstract
    # 1 introduction
    # 2 related work
    # 3 method
    # 4 graph and figure
    # 5 result
    # 6 others
    def get_referenced_place_num(self, paper_id):
        title = self.query_result.get("title", "")
        # 从xml中检索到序号
        if self.tree is None:
            return [0 * 8]

        paper_number = -1
        self.paper_number = paper_number
        # print("paper_number", self.paper_number)
        for i, item in enumerate(self.biblStruct):
            this_test = item.xpath('.//*[local-name()="title"]')
            this_text = this_test[0].text
            if this_text is None:
                try:
                    this_text = this_test[1].text
                except IndexError:
                    this_text = ""
            try:
                score = fuzz.partial_ratio(title, this_text)
            except ValueError:
                score = 0
            if score >= 80:
                paper_number = i + 1
                break
        place_num = [0 for i in range(8)]

        self.paper_number = paper_number
        if paper_number == -1:
            return place_num
        # 使用序号，在xml文件中检索位置
        nodes = self.tree.xpath(f"//*[contains(text(), '[{paper_number}]')]")
        reference_times = len(nodes)

        for item in nodes:
            found_text = ""
            this_node = item
            while found_text == "":
                this_node = this_node.getparent()
                if this_node is None:
                    break
                if this_node.xpath("local-name()") == "figure":
                    place_num[4] = 1
                it_children = this_node.iterchildren()
                for jtem in it_children:
                    node = this_node
                    if jtem.xpath("local-name()") == "head":
                        found_text = node.text
                        n_num = jtem.attrib.get("n")
                        node = this_node
                        if n_num is None:
                            break
                        while not n_num.isdigit():
                            node = node.getprevious()
                            if node is None:
                                break
                            node_children = node.iterchildren()
                            for ktem in node_children:
                                if ktem.xpath("local-name()") == "head":
                                    n = ktem.attrib.get("n")
                                    if n is not None and n.isdigit():
                                        n_num = ktem.attrib.get("n")
                                        found_text = ktem.text
                                        break
                                    break

            if this_node is None or found_text == "":
                place_num[6] = 1
                continue
            if found_text is not None:
                found_text = found_text.lower()
            if fuzz.partial_ratio("abstract", found_text) >= 60:
                place_num[0] = 1
            elif fuzz.partial_ratio("introduction", found_text) >= 60:
                place_num[1] = 1
            elif fuzz.partial_ratio("related work", found_text) >= 60:
                place_num[2] = 1
            elif fuzz.partial_ratio("method", found_text) >= 60:
                place_num[3] = 1
            elif (
                fuzz.partial_ratio("result", found_text) >= 60
                or fuzz.partial_ratio("experiment", found_text) >= 60
            ):
                place_num[5] = 1
            else:
                place_num[6] = 1
        pattern = re.compile(r"[\d+]")
        nodes = self.tree.xpath(
            "//*[re:match(text(), $pattern)]",
            namespaces={"re": "http://exslt.org/regular-expressions"},
            pattern=pattern.pattern,
        )
        total_ref_num = len(nodes)
        if not total_ref_num == 0:
            place_num[7] = reference_times / total_ref_num
        return place_num

    # FOUR 重叠作者
    def get_common_authors(self, paper_id):
        ref_authors_set = set(
            [item.get("name") for item in self.query_result.get("authors", "")]
        )
        # ref_authors_set = set(self.query_result.get("authors", ""))
        if not len(self.authors_set & ref_authors_set) == 0:
            return 1
        else:
            return 0

    # FIVE 关键词
    def key_words(self):
        if self.paper_number == -1:
            return 0
        pattern = re.compile(r"[\d+]")
        nodes = self.tree.xpath(
            "//*[re:match(text(), $pattern)]",
            namespaces={"re": "http://exslt.org/regular-expressions"},
            pattern=pattern.pattern,
        )
        key_words_list = ["motivated by", "inspired by"]
        for item in nodes:
            if item.xpath("local-name()") == "ref":
                node_text = item.getparent().text
            else:
                node_text = item.text
            if node_text is None:
                return 0
            node_text = node_text.lower()
            for jtem in key_words_list:
                pattern = re.compile(rf"{jtem}")
                match = pattern.search(node_text)
                if match is not None:
                    return 1
        return 0

    # SEVEN

    def get_reciprocal_of_reference_num(self):
        if self.num_ref == 0:
            return 0
        else:
            return 1 / self.num_ref


def extract_train_features():
    data_dir = join(DATA_TRACE_DIR, "PST")
    with open(
        join(data_dir, "paper_source_trace_train_ans.json"), "r", encoding="utf-8"
    ) as read_file:
        data_dic = json.load(read_file)
    all_id = [item["_id"] for item in data_dic]
    data_list = []
    label_list = []
    item_list = []
    ref_pid_list = []
    year_list = []

    paper_info_more = load_json("data", "paper_info_hit_from_dblp_test_pub.json")

    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        process_data = Process_data(data_dic[i], "train", paper_info_more)
        this_data, this_label, ref_pids = (
            process_data.data_list,
            process_data.label_list,
            process_data.ref_pid_list,
        )
        data_list += this_data
        label_list += this_label
        ref_pid_list += ref_pids
        item_list += [item for _ in range(len(this_data))]

    df_feature = pd.DataFrame(data_list, columns=process_data.columns)
    df_feature["pid"] = item_list
    df_feature["label"] = [label[0] for label in label_list]
    df_feature["ref_pid"] = ref_pid_list

    df_feature = df_feature.drop_duplicates(
        ["pid", "ref_pid"], keep="first"
    ).reset_index(drop=True)

    return df_feature


def extract_valid_features(mode_df=True):
    data_dir = join(DATA_TRACE_DIR, "PST")
    with open(
        join(data_dir, "paper_source_trace_valid_wo_ans.json"), "r", encoding="utf-8"
    ) as read_file:
        data_dic = json.load(read_file)
    all_id = [item["_id"] for item in data_dic]
    # paper_info_more = load_json("../dataset/", "paper_info_hit_from_dblp.json")
    paper_info_more = load_json("data", "paper_info_hit_from_dblp_test_pub.json")

    total_data_dic = {}
    total_data = []
    item_list = []
    ref_pid_list = []
    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        process_data = Process_data(data_dic[i], "test", paper_info_more)
        n_refs = len(data_dic[i].get("references", []))
        this_data, this_label, ref_pids = (
            process_data.data_list,
            process_data.label_list,
            process_data.ref_pid_list,
        )

        item_list += [item for _ in range(len(this_data))]
        ref_pid_list += ref_pids

        if mode_df:
            total_data += this_data
        else:
            total_data_dic[item] = this_data

        assert len(this_data) == n_refs

        # if i > 10:
        #    break

    print("total_data", len(total_data))
    # print(pd.DataFrame(total_data))

    if mode_df:
        df_test = pd.DataFrame(total_data, columns=process_data.columns)
        df_test["pid"] = item_list
        df_test["ref_pid"] = ref_pid_list

        df_test = df_test.drop_duplicates(["pid", "ref_pid"], keep="first").reset_index(
            drop=True
        )

        return df_test
    else:
        return total_data_dic


def extract_test_pub_features(mode_df=True):
    data_dir = join(DATA_TRACE_DIR, "PST")
    with open(
        join(data_dir, "paper_source_trace_test_wo_ans.json"), "r", encoding="utf-8"
    ) as read_file:
        data_dic = json.load(read_file)
    all_id = [item["_id"] for item in data_dic]
    # paper_info_more = load_json("../dataset/", "paper_info_hit_from_dblp.json")
    paper_info_more = load_json("../data", "paper_info_hit_from_dblp_test_pub.json")

    total_data_dic = {}
    total_data = []
    item_list = []
    ref_pid_list = []
    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        process_data = Process_data(data_dic[i], "test", paper_info_more)
        n_refs = len(data_dic[i].get("references", []))
        this_data, this_label, ref_pids = (
            process_data.data_list,
            process_data.label_list,
            process_data.ref_pid_list,
        )

        item_list += [item for _ in range(len(this_data))]
        ref_pid_list += ref_pids

        if mode_df:
            total_data += this_data
        else:
            total_data_dic[item] = this_data

        assert len(this_data) == n_refs

        # if i > 10:
        #    break

    print("total_data", len(total_data))
    # print(pd.DataFrame(total_data))

    if mode_df:
        df_test = pd.DataFrame(total_data, columns=process_data.columns)
        df_test["pid"] = item_list
        df_test["ref_pid"] = ref_pid_list

        df_test = df_test.drop_duplicates(["pid", "ref_pid"], keep="first").reset_index(
            drop=True
        )

        return df_test
    else:
        return total_data_dic


def main():
    os.makedirs("../data", exist_ok=True)
    df_test_pub_dblp = extract_test_pub_features()
    df_test_pub_dblp.to_csv("../data/df_test_pub_dblp.csv", index=False)
    df_train_dblp = extract_train_features()
    df_train_dblp.to_csv("../data/df_train_dblp.csv", index=False)
    df_test_dblp = extract_valid_features()
    df_test_dblp.to_csv("../data/df_test_dblp.csv", index=False)


if __name__ == "__main__":
    main()
