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
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

from sentence_transformers import InputExample
from torch.utils.data import DataLoader

import math
from collections import OrderedDict

NG_WORD_LIST = [
    "No abstract available.  ",
    'Use the "Report an Issue" link to request a name change.',
    "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000",
    "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.",
]

context_pattern = re.compile(r'<ref[^>]*type="bibr"[^>]*>(.*?)</ref>')
head_pattern = re.compile(r"<head.*?</head>")
div_pattern = re.compile(r"<div.*?</div>")
div_pattern2 = re.compile(r"<div.*?</div>")
formula_pattern = re.compile(r"<formula.*?</formula>")
ref_pattern = re.compile(r"<div.*?>")
multi_comma_pattern = re.compile(r",+")
multi_space_pattern = re.compile(r"\s+")


def load_json(rfdir, rfname):
    print("loading %s ...", rfname)
    with open(join(rfdir, rfname), "r", encoding="utf-8") as rf:
        data = json.load(rf)
        print("%s loaded", rfname)
        return data


def find_bib_context(xml, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = OrderedDict()
    bibr_strs_to_bid_id = OrderedDict()
    _tags = bs.find_all(type="bibr")
    sorted_tags = sorted(_tags, key=lambda x: len(x.get_text()), reverse=True)
    for item in sorted_tags:
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]
        item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
            item.attrs["target"], item.get_text()
        )
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [
            ii for ii in range(len(xml)) if xml.startswith(item_str, ii)
        ]
        for pos in cur_bib_context_pos_start:
            if bib_id not in bib_to_context:
                bib_to_context[bib_id] = []
            bib_to_context[bib_id].append(
                xml[pos - dist : pos + dist + len(item_str)]
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )
    return bib_to_context


def find_bib_context_2(xml, cur_pid, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = OrderedDict()
    bibr_strs_to_bid_id = OrderedDict()
    _tags = bs.find_all(type="bibr")
    sorted_tags = sorted(_tags, key=lambda x: len(x.get_text()), reverse=True)

    ref_ptn = re.compile(r"^\(\d+\)$")

    for item in sorted_tags:
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]

        ref_parentheses = re.findall(ref_ptn, item.get_text())
        if len(ref_parentheses) > 0:
            _ref = ref_parentheses[0]

            # (2020)の場合があるのでそれはスキップ
            # if int(_ref.lstrip("(").rstrip(")")) < 1990:
            if int(_ref[1:-1]) < 1990:
                # print(cur_pid, bib_id)
                # print(item.get_text())
                continue

        item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
            item.attrs["target"], item.get_text()
        )
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [
            ii for ii in range(len(xml)) if xml.startswith(item_str, ii)
        ]
        for pos in cur_bib_context_pos_start:
            if bib_id not in bib_to_context:
                bib_to_context[bib_id] = []
            bib_to_context[bib_id].append(
                xml[pos - dist : pos + dist + len(item_str)]
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )
    return bib_to_context


def find_bib_context_3(xml, cur_pid, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = OrderedDict()
    bibr_strs_to_bid_id = OrderedDict()
    _tags = bs.find_all(type="bibr")
    sorted_tags = sorted(_tags, key=lambda x: len(x.get_text()), reverse=True)

    ref_ptn = re.compile(r"^\(\d+\)$")

    for item in sorted_tags:
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]

        ref_parentheses = re.findall(ref_ptn, item.get_text())
        if len(ref_parentheses) > 0:
            _ref = ref_parentheses[0]

            # (2020)の場合があるのでそれはスキップ
            # if int(_ref.lstrip("(").rstrip(")")) < 1990:
            if int(_ref[1:-1]) < 1990:
                # print(cur_pid, bib_id)
                # print(item.get_text())
                continue

        item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
            item.attrs["target"], item.get_text()
        )
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [
            ii for ii in range(len(xml)) if xml.startswith(item_str, ii)
        ]
        for pos in cur_bib_context_pos_start:
            if bib_id not in bib_to_context:
                bib_to_context[bib_id] = []
            bib_to_context[bib_id].append(
                xml[pos - dist : pos + len(item_str)]
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )
    return bib_to_context


def find_bib_context_4(xml, cur_pid, dist=150, clean=False):

    def _parse_context(xml, pos, dist, item_str, replace_context="the reference paper"):
        start = pos - dist
        end = pos + dist + len(item_str)

        context = xml[start:end].replace("\n", " ").replace("\r", " ").strip()

        context = context.replace("</p>", "")
        context = context.replace("<p>", "")
        context = context.replace("<p></p>", "")

        if clean:
            context = context.replace(item_str, replace_context)
            context = re.sub(context_pattern, "", context)
            context = re.sub(head_pattern, "", context)
            context = re.sub(div_pattern, "", context)
            context = re.sub(ref_pattern, "", context)
            context = re.sub(div_pattern2, "", context)

        context = re.sub(formula_pattern, "", context)

        context = context.replace("</div>", "")
        context = context.replace("</head>", "")
        context = context.replace("</formula>", "")
        context = context.replace("</ref>", "")
        context = context.replace("( )", "")
        context = context.replace("?", "")

        context = re.sub(multi_comma_pattern, "", context)
        context = re.sub(multi_space_pattern, " ", context)

        return context

    bs = BeautifulSoup(xml, "xml")
    bib_to_context = OrderedDict()
    bibr_strs_to_bid_id = OrderedDict()
    _tags = bs.find_all(type="bibr")
    sorted_tags = sorted(_tags, key=lambda x: len(x.get_text()), reverse=True)

    ref_ptn = re.compile(r"^\(\d+\)$")
    item_ptn = re.compile(r"<ref[^>]*>(.*?)</ref>")

    for item in sorted_tags:
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]

        ref_parentheses = re.findall(ref_ptn, item.get_text())
        if len(ref_parentheses) > 0:
            _ref = ref_parentheses[0]

            # (2020)の場合があるのでそれはスキップ
            # if int(_ref.lstrip("(").rstrip(")")) < 1990:
            if int(_ref[1:-1]) < 1990:
                # print(cur_pid, bib_id)
                # print(item.get_text())
                continue

        # item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
        #     item.attrs["target"], item.get_text()
        # )
        # item_str = '<ref type="bibr" target="{}">'.format(item.attrs["target"])

        if "&amp;" in item.__str__():
            text = item.__str__()

            # pattern = r"<ref[^>]*>(.*?)</ref>"

            _item = re.findall(item_ptn, text)

            item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
                item.attrs["target"], _item[0]
            )
        else:

            item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
                item.attrs["target"], item.get_text()
            )

        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [
            ii for ii in range(len(xml)) if xml.startswith(item_str, ii)
        ]
        for pos in cur_bib_context_pos_start:
            if bib_id not in bib_to_context:
                bib_to_context[bib_id] = []

            # for _ in range(3):
            #     context = _parse_context(xml, pos, dist, item_str)
            #     dist += 10
            #     if len(context) > len("the reference paper"):
            #         break
            context = _parse_context(xml, pos, dist * 2, item_str)

            if len(context) > dist * 3:
                context = context[dist // 2 : -(dist // 2)]

            bib_to_context[bib_id].append(context)

    return bib_to_context


def find_bib_context_cleasing(xml, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = OrderedDict()
    bibr_strs_to_bid_id = OrderedDict()
    _tags = bs.find_all(type="bibr")
    sorted_tags = sorted(_tags, key=lambda x: len(x.get_text()), reverse=True)
    for item in sorted_tags:
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]
        item_str = '<ref type="bibr" target="{}">{}</ref>'.format(
            item.attrs["target"], item.get_text()
        )
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [
            ii for ii in range(len(xml)) if xml.startswith(item_str, ii)
        ]
        for pos in cur_bib_context_pos_start:
            if bib_id not in bib_to_context:
                bib_to_context[bib_id] = []

            s = (
                xml[pos - dist : pos + dist + len(item_str)]
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )

            bib_to_context[bib_id].append(s)

    return bib_to_context


def get_cur_paper_title(paper_meta_data, cur_pid, bs):
    """
    Get the title of the current paper
    """
    if cur_pid in paper_meta_data:
        cur_title = paper_meta_data[cur_pid]["title"]  # .lower()
    else:
        _title = ""
        if bs.find("titleStmt") is not None:
            _title = (
                bs.find("titleStmt")
                .title.get_text()
                .replace("\n", "")
                .replace("\r", "")
            )

        cur_title = _title

    if cur_pid == "617d22045244ab9dcbd01de7":
        cur_title = list(
            bs.find_all("body")[0].find_all(
                "div", {"xmlns": "http://www.tei-c.org/ns/1.0"}
            )
        )[1].get_text()

    return cur_title


def create_bid_to_title_master(bs):
    references = bs.find_all("biblStruct")
    bid_to_title = OrderedDict()
    n_refs = 0
    ## xmlパース(analitic.title)
    for ref in references:
        if "xml:id" not in ref.attrs:
            continue
        bid = ref.attrs["xml:id"]
        if ref.analytic is None:
            continue
        if ref.analytic.title is None:
            continue
        bid_to_title[bid] = ref.analytic.title.text.lower()
        b_idx = int(bid[1:]) + 1
        if b_idx > n_refs:
            n_refs = b_idx

    ## xmlパース(title)
    for ref in references:
        if "xml:id" not in ref.attrs:
            continue
        bid = ref.attrs["xml:id"]
        b_idx = int(bid[1:])
        if b_idx >= n_refs:
            continue
        if ref.title is not None and bid not in bid_to_title:
            bid_to_title[bid] = ref.title.text.lower()

        if ref.title is None and bid not in bid_to_title:
            print(ref)

    return bid_to_title, n_refs


def create_pos_neg_bib_list(
    cur_pid, pid_to_source_titles, bid_to_title, n_negative_sample=10
):
    flag = False
    cur_pos_bib = []
    source_titles = pid_to_source_titles[cur_pid]
    if len(source_titles) == 0:
        return []
    for bid in bid_to_title:
        cur_ref_title = bid_to_title[bid]
        for label_title in source_titles:
            if fuzz.ratio(cur_ref_title, label_title) >= 80:
                flag = True
                if bid not in cur_pos_bib:
                    cur_pos_bib.append(bid)

    cur_neg_bib = [bid for bid in bid_to_title.keys() if bid not in cur_pos_bib]

    if not flag:
        return []

    if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
        return []

    n_pos = len(cur_pos_bib)
    if n_negative_sample > 0:
        n_neg = n_pos * n_negative_sample
        np.random.seed(42)
        cur_neg_bib_sample = list(np.random.choice(cur_neg_bib, n_neg, replace=True))
    else:
        cur_neg_bib_sample = cur_neg_bib

    bib_list = cur_pos_bib + cur_neg_bib_sample
    label_list = [1 for _ in range(len(cur_pos_bib))] + [
        0 for _ in range(len(cur_neg_bib_sample))
    ]
    return bib_list, label_list


def create_bid_to_pid(bid_to_title, paper, paper_meta_data, in_dir):
    # def check_ref(ref):
    #     if "xml:id" not in ref.attrs:
    #         return False
    #     if ref.analytic is None:
    #         return False
    #     if ref.analytic.title is None:
    #         return False

    #     return True

    # def check_ref_2(ref):
    #     if "xml:id" not in ref.attrs:
    #         return False
    #     if ref.title is None:
    #         return False

    #     return True

    # references = bs.find_all("biblStruct")
    # references_xml = []
    references_master = []
    bid_pid_dict = {}

    # for ref in references:
    #     if check_ref(ref):
    #         references_xml.append(
    #             (ref.attrs["xml:id"], ref.analytic.title.text.lower())
    #         )
    #     elif check_ref_2(ref):
    #         references_xml.append((ref.attrs["xml:id"], ref.title.text.lower()))
    #     else:
    #         continue

    references_xml = [(k, v) for k, v in bid_to_title.items()]

    for pid in paper["references"]:
        if pid in paper_meta_data:
            references_master.append((pid, paper_meta_data[pid]["title"].lower()))
        else:
            try:
                ref_file = join(in_dir, pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            if bs.find("titleStmt") is not None:
                _title = (
                    bs.find("titleStmt")
                    .title.get_text()
                    .lower()
                    .replace("\n", "")
                    .replace("\r", "")
                )
                references_master.append((pid, _title))
            else:
                continue

    for bid, ref_xml in references_xml:
        for pid, ref_master in references_master:
            if fuzz.token_sort_ratio(ref_xml, ref_master) > 90:
                bid_pid_dict[bid] = pid
                # master.append([cur_pid, pid, ref_master, bid, ref_xml])

    return bid_pid_dict


def is_chinese(s):
    for c in s:
        if "\u4e00" <= c <= "\u9fa5":
            return True
    return False


def get_cur_paper_abst(paper_meta_data, cur_pid, bs):
    """
    Get the abstract of the current paper
    """

    cur_abstract = ""
    if cur_pid in paper_meta_data:
        cur_abstract = (
            paper_meta_data[cur_pid]["abstract"].replace("\n", "").replace("\r", "")
        )

    if cur_abstract == "" or cur_abstract in NG_WORD_LIST:

        if bs.abstract is not None:
            cur_abstract = bs.abstract.get_text().replace("\n", "").replace("\r", "")
        else:
            cur_abstract = (
                bs.find("abstract").get_text().replace("\n", "").replace("\r", "")
            )

    if cur_abstract in NG_WORD_LIST:
        cur_abstract = bs.body.get_text().replace("\n", "").replace("\r", "")[:256]

    # if is_chinese(cur_abstract):
    #    cur_abstract = ""

    return cur_abstract


def get_cur_org(paper_meta_data, cur_pid, bs):
    cur_org = []

    if len(cur_org) == 0:
        if bs.find_all("orgName") is not None:
            cur_org = list(set([x.text for x in bs.find_all("orgName")]))

    return cur_org


def get_cur_year(paper_meta_data, cur_pid, bs):
    cur_year = None
    if cur_pid in paper_meta_data:
        cur_year = int(paper_meta_data[cur_pid]["year"])
    if cur_year is None:
        if bs.find_all("date") is not None:

            years = [re.findall(r"\b\d{4}\b", x.text) for x in bs.find_all("date")]
            years = [max(year) for year in years if len(year) > 0]
            if len(years) > 0:
                cur_year = int(max(years))

        else:
            cur_year = None

    return cur_year


def get_cur_venue(paper_meta_data, cur_pid, bs):
    cur_venue = ""
    if cur_pid in paper_meta_data:
        if paper_meta_data[cur_pid] is not None:
            try:
                cur_venue = paper_meta_data[cur_pid].get("venue", "")
                if cur_venue is None:
                    cur_venue = ""
                else:
                    cur_venue = cur_venue.replace("\n", "").replace("\r", "").lower()
            except:
                print(cur_pid, cur_venue)
                raise ValueError()

    # remove url
    cur_venue = re.sub(
        r"(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?", "", cur_venue
    )
    return cur_venue


def get_cur_paper_keyword(paper_meta_data, cur_pid, bs):
    cur_keyword = []
    if cur_pid in paper_meta_data:
        cur_keyword = paper_meta_data[cur_pid]["keywords"]
        cur_keyword = [x.replace("\n", "").replace("\r", "") for x in cur_keyword]

        # " ".join(paper_meta_data[cur_pid]["keywords"])
        # .replace("\n", "")
        # .replace("\r", "")

    if len(cur_keyword) == 0:
        if bs.keywords is not None:
            cur_keyword = [bs.keywords.get_text().replace("\n", "").replace("\r", "")]
        elif bs.find("keywords") is not None:
            cur_keyword = [
                bs.find("keywords").get_text().replace("\n", "").replace("\r", "")
            ]
        else:
            pass

    return cur_keyword


def get_cur_paper_org(paper_meta_data, cur_pid, bs):
    cur_org = []
    if cur_pid in paper_meta_data:
        authros = paper_meta_data[cur_pid].get("authors", "")
        if authros is not None:
            orgs = []
            for author_dict in authros:
                org = author_dict.get("org", "")
                if (org is not None) and (org != "") and org not in orgs:
                    orgs.append(org)

            # cur_org = " ".join(orgs).replace("\n", "").replace("\r", "").lower()
            cur_org = [
                org.replace("\n", "").replace("\r", "") for org in orgs if org != ""
            ]

    if len(cur_org) == 0:
        if bs.find_all("orgName") is not None:
            orgs = sorted([x.text for x in bs.find_all("orgName")])
            _orgs = []
            for org in _orgs:
                if org not in _orgs:
                    _orgs.append(org)

            cur_org = [
                org.replace("\n", "").replace("\r", "") for org in _orgs if org != ""
            ]

    return cur_org


def get_cur_paper_authors(paper_meta_data, cur_pid, bs):
    cur_authors = []
    if cur_pid in paper_meta_data:
        authros = paper_meta_data[cur_pid].get("authors", "")
        if authros is not None:
            authors_list = []
            for author_dict in authros:
                name = author_dict.get("name", "")
                if (name is not None) and (name != "") and name not in authors_list:
                    authors_list.append(name)

            cur_authors = authors_list

    if len(cur_authors) == 0:
        if bs.find_all("persName") is not None:

            for pers in bs.find_all("persName"):
                forename = pers.find("forename")
                surname = pers.find("surname")
                if forename is not None and surname is not None:
                    name = "{} {}".format(forename.text, surname.text)
                    cur_authors.append(name)

    return cur_authors


def get_cur_paper_doi(paper_meta_data, cur_pid, bs):
    cur_doi = ""
    if cur_pid in paper_meta_data:
        cur_doi = paper_meta_data[cur_pid].get("doi", "")
    if cur_doi == "":
        if bs.find("idno", type="DOI") is not None:
            cur_doi = bs.find("idno", type="DOI").text
    return cur_doi


def extract_preceding_text_with_regex(text, pattern, num_chars=200):
    # 正規表現パターンをコンパイル
    regex = re.compile(pattern)

    # 正規表現パターンにマッチする部分を探す
    match = regex.search(text)

    # パターンが見つからない場合
    if not match:
        # print(f"pattern '{pattern}' not found")
        return ""

    # マッチした部分の開始位置を取得
    position = match.start()

    # 前の200文字を抽出する
    start_position = max(position - num_chars, 0)
    preceding_text = text[start_position:position]

    return preceding_text


def extract_preceding_text_with_regex_from(text, pattern, num_chars=200):
    # 正規表現パターンをコンパイル
    regex = re.compile(pattern)

    # 正規表現パターンにマッチする部分を探す
    match = regex.search(text)

    # パターンが見つからない場合
    if not match:
        # print(f"pattern '{pattern}' not found")
        return ""

    # マッチした部分の開始位置を取得
    position = match.start()

    last_position = min(position + num_chars, len(text))

    preceding_text = text[position:last_position]

    return preceding_text


def get_cur_paper_intoroduction_last(paper_meta_data, cur_pid, bs, dist=200):
    cur_introduction = ""

    if len(bs.select('div:contains("Introduction")')) > 0:
        cur_introduction = (
            bs.select('div:contains("Introduction")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )
    elif len(bs.select('div:contains("INTRODUCTION")')) > 0:
        cur_introduction = (
            bs.select('div:contains("INTRODUCTION")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )

    # sectionという文字列が含まれる場合、sectionの前を取得する
    # section 1 hogehoge, section 2 hogehoge などが含まれるのでsectionという文字列の前を取得する
    if "section" in cur_introduction[-dist:] or "Section" in cur_introduction[-dist:]:
        cur_introduction = extract_preceding_text_with_regex(
            cur_introduction, r"Section|section", 300
        )

    cur_introduction = cur_introduction[-dist:]

    return cur_introduction


def get_cur_paper_intoroduction_from(paper_meta_data, cur_pid, bs, dist=200):
    cur_introduction = ""

    if len(bs.select('div:contains("Introduction")')) > 0:
        cur_introduction = (
            bs.select('div:contains("Introduction")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )
    elif len(bs.select('div:contains("INTRODUCTION")')) > 0:
        cur_introduction = (
            bs.select('div:contains("INTRODUCTION")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )

    # sectionという文字列が含まれる場合、sectionの前を取得する
    # section 1 hogehoge, section 2 hogehoge などが含まれるのでsectionという文字列の前を取得する

    if "we propose" in cur_introduction or "We propose" in cur_introduction:
        cur_introduction = extract_preceding_text_with_regex_from(
            cur_introduction, r"we propose|We propose", 300
        )

    elif "section" in cur_introduction[-dist:] or "Section" in cur_introduction[-dist:]:
        cur_introduction = extract_preceding_text_with_regex(
            cur_introduction, r"Section|section", 300
        )

    return cur_introduction


def get_cur_paper_conclusion_begin(paper_meta_data, cur_pid, bs, dist=200):
    cur_conclusion = ""

    if len(bs.select('div:contains("Conclusion")')) > 0:
        cur_conclusion = (
            bs.select('div:contains("Conclusion")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]
    elif len(bs.select('div:contains("CONCLUSION")')) > 0:
        cur_conclusion = (
            bs.select('div:contains("CONCLUSION")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]

    elif len(bs.select('div:contains("CONCLUSIONS")')) > 0:
        cur_conclusion = (
            bs.select('div:contains("CONCLUSIONS")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]
    elif len(bs.select('div:contains("Conclusions")')) > 0:
        cur_conclusion = (
            bs.select('div:contains("Conclusions")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]

    return cur_conclusion


def get_cur_paper_related_work_begin(paper_meta_data, cur_pid, bs, dist=200):
    cur_related_work = ""

    if len(bs.select('div:contains("Related Work")')) > 0:
        cur_related_work = (
            bs.select('div:contains("Related Work")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]
    elif len(bs.select('div:contains("RELATED WORK")')) > 0:
        cur_related_work = (
            bs.select('div:contains("RELATED WORK")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]
    elif len(bs.select('div:contains("Related work")')) > 0:
        cur_related_work = (
            bs.select('div:contains("Related work")')[0]
            .get_text()
            .replace("\n", "")
            .replace("\r", "")
        )[:dist]

    return cur_related_work


def get_ref_pid_abst(paper_meta_data, paper, in_dir):
    """
    Get the abstract of the reference paper
    """

    abstracts = {}
    for ref_pid in paper["references"]:
        abstracts[ref_pid] = ""
        if ref_pid in paper_meta_data:
            abstracts[ref_pid] = (
                paper_meta_data[ref_pid]["abstract"].replace("\n", "").replace("\r", "")
            )

        if abstracts[ref_pid] == "" or (abstracts[ref_pid] in NG_WORD_LIST):

            if os.path.exists(join(in_dir, ref_pid + ".xml")):
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
                abstracts[ref_pid] = get_cur_paper_abst(paper_meta_data, ref_pid, bs)

                if abstracts[ref_pid] == "" or abstracts[ref_pid] in NG_WORD_LIST:
                    abstracts[ref_pid] = (
                        bs.body.get_text().replace("\n", "").replace("\r", "")[:256]
                    )

        if abstracts[ref_pid] in NG_WORD_LIST:
            abstracts[ref_pid] = ""

    return abstracts


def get_ref_pid_keyword(paper_meta_data, paper, in_dir):
    """
    Get the keywords of the reference paper
    """

    keywords = {}
    for ref_pid in paper["references"]:
        keywords[ref_pid] = ""
        if ref_pid in paper_meta_data:
            _kws = paper_meta_data[ref_pid]["keywords"]
            keywords[ref_pid] = [x.replace("\n", "").replace("\r", "") for x in _kws]

        if len(keywords.get(ref_pid, [])) == 0:
            try:
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            keywords[ref_pid] = get_cur_paper_keyword(paper_meta_data, ref_pid, bs)

    return keywords


def get_ref_pid_year(paper_meta_data, paper, in_dir):
    years = {}
    for ref_pid in paper["references"]:
        if ref_pid in paper_meta_data:
            years[ref_pid] = paper_meta_data[ref_pid].get("year", None)
        if ref_pid not in years or years[ref_pid] is None:
            try:
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            years[ref_pid] = get_cur_year(paper_meta_data, ref_pid, bs)
    return years


def get_ref_pid_venue(paper_meta_data, paper, in_dir):
    venues = {}
    for ref_pid in paper["references"]:
        if ref_pid in paper_meta_data:
            venues[ref_pid] = paper_meta_data[ref_pid].get("venue", "")
            if venues[ref_pid] is None:
                venues[ref_pid] = ""
            else:
                venues[ref_pid] = (
                    venues[ref_pid].replace("\n", "").replace("\r", "").lower()
                )

            # remove url
            venues[ref_pid] = re.sub(
                r"(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?", "", venues[ref_pid]
            )

    return venues


def get_ref_pid_paper_org(paper_meta_data, paper, in_dir):

    orgs = {}
    for ref_pid in paper["references"]:
        if ref_pid in paper_meta_data:
            authros = paper_meta_data[ref_pid].get("authors", "")
            _orgs = []
            for author_dict in authros:
                org = author_dict.get("org", "")
                if (org is not None) and (org != "") and org not in _orgs:
                    _orgs.append(org)

            orgs[ref_pid] = [org.replace("\n", "").replace("\r", "") for org in _orgs]

        if len(orgs.get(ref_pid, [])) == 0:
            try:
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            orgs[ref_pid] = get_cur_paper_org(paper_meta_data, ref_pid, bs)

    return orgs


def get_ref_pid_authors(paper_meta_data, paper, in_dir):
    authors = {}
    for ref_pid in paper["references"]:
        if ref_pid in paper_meta_data:
            authros = paper_meta_data[ref_pid].get("authors", "")
            authors_list = []
            for author_dict in authros:
                name = author_dict.get("name", "")
                if (name is not None) and (name != "") and name not in authors_list:
                    authors_list.append(name)

            authors[ref_pid] = authors_list

        if len(authors.get(ref_pid, [])) == 0:
            try:
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            authors[ref_pid] = get_cur_paper_authors(paper_meta_data, ref_pid, bs)

    return authors


def get_ref_pid_doi(paper_meta_data, paper, in_dir):
    dois = {}
    for ref_pid in paper["references"]:
        if ref_pid in paper_meta_data:
            dois[ref_pid] = paper_meta_data[ref_pid].get("doi", "")
        if ref_pid not in dois or dois.get(ref_pid, "") == "":
            try:
                ref_file = join(in_dir, ref_pid + ".xml")
                ref_f = open(ref_file, encoding="utf-8")
                xml = ref_f.read()
                bs = BeautifulSoup(xml, "xml")
                ref_f.close()
            except FileNotFoundError:
                continue
            dois[ref_pid] = get_cur_paper_doi(paper_meta_data, ref_pid, bs)
    return dois


def get_ref_pid_intoroduction_last(paper_meta_data, paper, in_dir, dist=200):
    introductions = {}
    for ref_pid in paper["references"]:
        introductions[ref_pid] = ""
        if os.path.exists(join(in_dir, ref_pid + ".xml")):
            ref_file = join(in_dir, ref_pid + ".xml")
            ref_f = open(ref_file, encoding="utf-8")
            xml = ref_f.read()
            bs = BeautifulSoup(xml, "xml")
            ref_f.close()
            introductions[ref_pid] = get_cur_paper_intoroduction_last(
                paper_meta_data, ref_pid, bs, dist
            )

    return introductions


def get_ref_pid_intoroduction_from(paper_meta_data, paper, in_dir, dist=200):
    introductions = {}
    for ref_pid in paper["references"]:
        introductions[ref_pid] = ""
        if os.path.exists(join(in_dir, ref_pid + ".xml")):
            ref_file = join(in_dir, ref_pid + ".xml")
            ref_f = open(ref_file, encoding="utf-8")
            xml = ref_f.read()
            bs = BeautifulSoup(xml, "xml")
            ref_f.close()
            # introductions[ref_pid] = get_cur_paper_intoroduction_last(
            #     paper_meta_data, ref_pid, bs, dist
            # )
            introductions[ref_pid] = get_cur_paper_intoroduction_from(
                paper_meta_data, ref_pid, bs, dist
            )

    return introductions


def get_ref_pid_conclusion_begin(paper_meta_data, paper, in_dir, dist=200):
    conclusions = {}
    for ref_pid in paper["references"]:
        conclusions[ref_pid] = ""
        if os.path.exists(join(in_dir, ref_pid + ".xml")):
            ref_file = join(in_dir, ref_pid + ".xml")
            ref_f = open(ref_file, encoding="utf-8")
            xml = ref_f.read()
            bs = BeautifulSoup(xml, "xml")
            ref_f.close()
            conclusions[ref_pid] = get_cur_paper_conclusion_begin(
                paper_meta_data, ref_pid, bs, dist
            )

    return conclusions


def get_ref_pid_related_work_begin(paper_meta_data, paper, in_dir, dist=200):
    related_works = {}
    for ref_pid in paper["references"]:
        related_works[ref_pid] = ""
        if os.path.exists(join(in_dir, ref_pid + ".xml")):
            ref_file = join(in_dir, ref_pid + ".xml")
            ref_f = open(ref_file, encoding="utf-8")
            xml = ref_f.read()
            bs = BeautifulSoup(xml, "xml")
            ref_f.close()
            related_works[ref_pid] = get_cur_paper_related_work_begin(
                paper_meta_data, ref_pid, bs, dist
            )

    return related_works


def prepare_bert_input_all(
    data_dir, hit_from_dblp_dir, debug=False, n_negative_sample=10, mode="train"
):

    assert mode in ["train", "test", "test_pub"]

    x_list = []
    context_clean_list = []
    y_list = []
    pid_list = []
    ref_title_list = []
    bid_list = []
    title_list = []
    ref_pid_list = []
    cur_abstract_list = []
    ref_abstract_list = []
    ref_count_list = []
    context_bibr_count_list = []
    context_target_bibr_count_list = []
    context_other_bibr_count_list = []
    cur_keywords_list = []
    ref_keywords_list = []

    cur_year_list = []
    ref_year_list = []
    cur_venue_list = []
    ref_venue_list = []

    cur_org_list = []
    ref_org_list = []

    cur_doi_list = []
    ref_doi_list = []

    cur_introduction_list = []
    ref_introduction_list = []

    cur_conclusion_list = []
    ref_conclusion_list = []

    cur_related_work_list = []
    ref_related_work_list = []

    cur_authors_list = []
    ref_authors_list = []

    if mode == "train":
        papers = load_json(data_dir, "paper_source_trace_train_ans.json")
    elif mode == "test":
        papers = load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    elif mode == "test_pub":
        papers = load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    paper_meta_data = load_json(
        hit_from_dblp_dir, "paper_info_hit_from_dblp_test_pub.json"
    )

    if debug:
        papers = papers[:200]

    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    if mode == "train":
        for paper in tqdm(papers):
            pid = paper["_id"]
            for ref in paper["refs_trace"]:
                pid_to_source_titles[pid].append(ref["title"].lower())

    for i, paper in tqdm(enumerate(papers), total=n_papers):
        cur_pid = paper["_id"]

        # 対象論文ファイルを読み込み
        f = open(join(in_dir, cur_pid + ".xml"), encoding="utf-8")
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        # 対象論文のタイトルを取得
        cur_title = get_cur_paper_title(paper_meta_data, cur_pid, bs)

        # abstractの取得
        cur_abstract = get_cur_paper_abst(paper_meta_data, cur_pid, bs)

        cur_keywords = get_cur_paper_keyword(paper_meta_data, cur_pid, bs)

        cur_year = get_cur_year(paper_meta_data, cur_pid, bs)
        cur_venue = get_cur_venue(paper_meta_data, cur_pid, bs)
        cur_org = get_cur_paper_org(paper_meta_data, cur_pid, bs)
        cur_doi = get_cur_paper_doi(paper_meta_data, cur_pid, bs)

        # Introduction
        cur_introduction = get_cur_paper_intoroduction_from(
            paper_meta_data, cur_pid, bs, dist=300
        )

        # Conclusion
        cur_conclusion = get_cur_paper_conclusion_begin(
            paper_meta_data, cur_pid, bs, dist=300
        )

        # Related work
        cur_related_work = get_cur_paper_related_work_begin(
            paper_meta_data, cur_pid, bs, dist=300
        )

        cur_authors = get_cur_paper_authors(paper_meta_data, cur_pid, bs)

        # 引用文章の取得
        # bib_to_contexts = find_bib_context(xml)
        # bib_to_contexts = find_bib_context_2(xml, cur_pid)
        # bib_to_contexts = find_bib_context_3(xml, cur_pid)
        bib_to_contexts = find_bib_context_4(xml, cur_pid)
        bib_to_contexts_clean = find_bib_context_4(xml, cur_pid, clean=True)

        # 正解データ作成のための、bid to title マスタの作成
        bid_to_title, n_refs = create_bid_to_title_master(bs)

        # 参照論文のPIDを取得
        bid_to_pid_dict = create_bid_to_pid(
            bid_to_title, paper, paper_meta_data, in_dir
        )

        # 参照論文のabstractを取得
        ref_pid_abst = get_ref_pid_abst(paper_meta_data, paper, in_dir)

        # 参照論文のkeywordを取得
        ref_pid_keyword = get_ref_pid_keyword(paper_meta_data, paper, in_dir)

        ref_pid_year = get_ref_pid_year(paper_meta_data, paper, in_dir)

        ref_pid_venue = get_ref_pid_venue(paper_meta_data, paper, in_dir)

        ref_pid_org = get_ref_pid_paper_org(paper_meta_data, paper, in_dir)
        ref_pid_doi = get_ref_pid_doi(paper_meta_data, paper, in_dir)

        # Introduction の最後の文章を取得
        ref_pid_introduction = get_ref_pid_intoroduction_from(
            paper_meta_data, paper, in_dir, dist=300
        )
        # Conclusion の最初の文章を取得
        ref_pid_conclusion = get_ref_pid_conclusion_begin(
            paper_meta_data, paper, in_dir, dist=300
        )

        # Related work の最初の文章を取得
        ref_pid_related_work = get_ref_pid_related_work_begin(
            paper_meta_data, paper, in_dir, dist=300
        )

        ref_pid_authors = get_ref_pid_authors(paper_meta_data, paper, in_dir)

        # データの結合と整形
        if mode == "train":
            # 正解と紐付け
            bib_list, label_list = create_pos_neg_bib_list(
                cur_pid, pid_to_source_titles, bid_to_title, n_negative_sample
            )
        else:
            bib_list = ["b" + str(ii) for ii in range(n_refs)]

        for bib in bib_list:
            ref_count = len(bib_to_contexts[bib]) if bib in bib_to_contexts else 1
            cur_context = (
                " ".join(bib_to_contexts[bib]) if bib in bib_to_contexts else ""
            )
            cur_context_clean = (
                " ".join(bib_to_contexts_clean[bib])
                if bib in bib_to_contexts_clean
                else ""
            )

            ref_pid = bid_to_pid_dict[bib] if bib in bid_to_pid_dict else ""
            ref_abst = ref_pid_abst[ref_pid] if ref_pid in ref_pid_abst else ""
            ref_keywords = (
                ref_pid_keyword[ref_pid] if ref_pid in ref_pid_keyword else ""
            )
            ref_year = ref_pid_year[ref_pid] if ref_pid in ref_pid_venue else None
            ref_venue = ref_pid_venue[ref_pid] if ref_pid in ref_pid_venue else ""

            ref_org = ref_pid_org[ref_pid] if ref_pid in ref_pid_org else ""
            ref_doi = ref_pid_doi[ref_pid] if ref_pid in ref_pid_doi else ""

            ref_introduction = (
                ref_pid_introduction[ref_pid] if ref_pid in ref_pid_introduction else ""
            )
            ref_conclusion = (
                ref_pid_conclusion[ref_pid] if ref_pid in ref_pid_conclusion else ""
            )

            ref_related_work = (
                ref_pid_related_work[ref_pid] if ref_pid in ref_pid_related_work else ""
            )

            ref_authors = ref_pid_authors[ref_pid] if ref_pid in ref_pid_authors else []

            x_list.append(cur_context)
            context_clean_list.append(cur_context_clean)
            pid_list.append(cur_pid)
            ref_title_list.append(bid_to_title[bib])
            bid_list.append(bib)
            title_list.append(cur_title)
            ref_pid_list.append(ref_pid)
            cur_abstract_list.append(cur_abstract)
            ref_abstract_list.append(ref_abst)
            ref_count_list.append(ref_count)
            context_bibr_count_list.append(cur_context.count("bibr"))
            context_target_bibr_count_list.append(cur_context.count(f"#{bib}"))

            context_other_bibr_count_list.append(
                cur_context.count(f"#b") - cur_context.count(f"#{bib}")
            )

            cur_keywords_list.append(cur_keywords)
            ref_keywords_list.append(ref_keywords)

            cur_year_list.append(cur_year)
            ref_year_list.append(ref_year)

            cur_venue_list.append(cur_venue)
            ref_venue_list.append(ref_venue)

            cur_org_list.append(cur_org)
            ref_org_list.append(ref_org)

            cur_doi_list.append(cur_doi)
            ref_doi_list.append(ref_doi)

            cur_introduction_list.append(cur_introduction)
            ref_introduction_list.append(ref_introduction)

            cur_conclusion_list.append(cur_conclusion)
            ref_conclusion_list.append(ref_conclusion)

            cur_related_work_list.append(cur_related_work)
            ref_related_work_list.append(ref_related_work)

            cur_authors_list.append(cur_authors)
            ref_authors_list.append(ref_authors)

        if mode == "train":
            y_list.extend(label_list)

        if debug and i > 200:
            break

    if mode == "train":
        return pd.DataFrame(
            {
                "pid": pid_list,
                "title": title_list,
                "ref_pid": ref_pid_list,
                "bid": bid_list,
                "ref_title": ref_title_list,
                "context": x_list,
                "context_clean": context_clean_list,
                "abstract": cur_abstract_list,
                "ref_abstract": ref_abstract_list,
                "xml_ref_count": ref_count_list,
                "context_bibr_count": context_bibr_count_list,
                "context_target_bibr_count": context_target_bibr_count_list,
                "context_other_bibr_count": context_other_bibr_count_list,
                "keywords": cur_keywords_list,
                "ref_keywords": ref_keywords_list,
                "year": cur_year_list,
                "ref_year": ref_year_list,
                "venue": cur_venue_list,
                "ref_venue": ref_venue_list,
                "org": cur_org_list,
                "ref_org": ref_org_list,
                "doi": cur_doi_list,
                "ref_doi": ref_doi_list,
                "introduction": cur_introduction_list,
                "ref_introduction": ref_introduction_list,
                "conclusion": cur_conclusion_list,
                "ref_conclusion": ref_conclusion_list,
                "related_work": cur_related_work_list,
                "ref_related_work": ref_related_work_list,
                "authors": cur_authors_list,
                "ref_authors": ref_authors_list,
                "label": y_list,
            }
        )

    else:

        return pd.DataFrame(
            {
                "pid": pid_list,
                "title": title_list,
                "ref_pid": ref_pid_list,
                "bid": bid_list,
                "ref_title": ref_title_list,
                "context": x_list,
                "context_clean": context_clean_list,
                "abstract": cur_abstract_list,
                "ref_abstract": ref_abstract_list,
                "xml_ref_count": ref_count_list,
                "context_bibr_count": context_bibr_count_list,
                "context_target_bibr_count": context_target_bibr_count_list,
                "context_other_bibr_count": context_other_bibr_count_list,
                "keywords": cur_keywords_list,
                "ref_keywords": ref_keywords_list,
                "year": cur_year_list,
                "ref_year": ref_year_list,
                "venue": cur_venue_list,
                "ref_venue": ref_venue_list,
                "org": cur_org_list,
                "ref_org": ref_org_list,
                "doi": cur_doi_list,
                "ref_doi": ref_doi_list,
                "introduction": cur_introduction_list,
                "ref_introduction": ref_introduction_list,
                "conclusion": cur_conclusion_list,
                "ref_conclusion": ref_conclusion_list,
                "related_work": cur_related_work_list,
                "ref_related_work": ref_related_work_list,
                "authors": cur_authors_list,
                "ref_authors": ref_authors_list,
            }
        )


def main():

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    debug = False

    df_test_pub_context = prepare_bert_input_all(
        data_dir="../dataset/PST/",
        hit_from_dblp_dir="data",
        debug=debug,
        mode="test_pub",
    )
    df_test_pub_context.to_csv(f"{output_dir}/test_pub_context.csv", index=False)

    df_train_context = prepare_bert_input_all(
        data_dir="../dataset/PST/",
        hit_from_dblp_dir="data",
        debug=debug,
        n_negative_sample=-1,
        mode="train",
    )
    print(df_train_context)

    # df_test_context = prepare_bert_input_all(
    #     data_dir="../dataset/PST/",
    #     hit_from_dblp_dir="data",
    #     debug=debug,
    #     mode="test",
    # )

    df_train_context.to_csv(f"{output_dir}/train_context.csv", index=False)
    # df_test_context.to_csv(f"{output_dir}/test_context.csv", index=False)

    df_train_context = pd.read_csv(f"{output_dir}/train_context.csv")
    # df_test_context = pd.read_csv(f"{output_dir}/test_context.csv")
    df_test_pub_context = pd.read_csv(f"{output_dir}/test_pub_context.csv")

    # print(df_test_context)

    # post process
    if debug:
        df_train_context.to_csv(f"{output_dir}/train_context.csv", index=False)
        # df_test_context.to_csv(f"{output_dir}/test_context.csv", index=False)

    # print("post process")
    # text = df_test_context[df_test_context["pid"] == "61cacea45244ab9dcb0ca982"][
    #     "abstract"
    # ].values[0]
    # text = text.split(".")[11]
    # df_test_context.loc[
    #     df_test_context["pid"] == "61cacea45244ab9dcb0ca982", ("title",)
    # ] = text
    # df_test_context.loc[
    #     df_test_context["pid"] == "61cacea45244ab9dcb0ca982", ("context",)
    # ].replace("?", "", inplace=True)

    # introduction からうめる
    text = df_train_context[df_train_context["pid"] == "5ce3a70cced107d4c652a6ae"][
        "introduction"
    ].values[0]
    text = text.split(".")[-2]
    df_train_context.loc[
        df_train_context["pid"] == "5ce3a70cced107d4c652a6ae", ("title",)
    ] = text

    text = df_train_context[df_train_context["pid"] == "5db9299b47c8f766461f9984"][
        "introduction"
    ].values[0]
    text = text.split(".")[-3]
    df_train_context.loc[
        df_train_context["pid"] == "5db9299b47c8f766461f9984", ("title",)
    ] = text

    # print("fill by oag")
    # df_train_oag = pd.read_csv(
    # n    f"../script/data_ochiai_san_2/df_train_context_filled_keywords.csv"
    # )
    # df_test_oag = pd.read_csv(
    #    f"../script/data_ochiai_san_2/df_test_context_filled_keywords.csv"
    # )

    df_train_context = pd.read_csv(f"{output_dir}/train_context.csv")

    # for col in [
    #     "title",
    #     "abstract",
    #     "keywords",
    #     "year",
    #     "venue",
    #     "org",
    #     "doi",
    #     "ref_title",
    #     "ref_abstract",
    #     "ref_keywords",
    #     "ref_year",
    #     "ref_venue",
    #     "ref_org",
    #     "ref_doi",
    # ]:
    #     print(
    #         "Before",
    #         col,
    #         df_train_context[col].isnull().sum(),
    #         # df_test_context[col].isnull().sum(),
    #     )
    #     df_train_context[col] = df_train_context[col].fillna(df_train_oag[col])
    #     # df_test_context[col] = df_test_context[col].fillna(df_test_oag[col])
    #     print(
    #         "After",
    #         col,
    #         df_train_context[col].isnull().sum(),
    #         # df_test_context[col].isnull().sum(),
    #     )

    #     df_train_context.to_csv(f"{output_dir}/train_context.csv", index=False)
    # df_test_context.to_csv(f"{output_dir}/test_context.csv", index=False)


if __name__ == "__main__":
    main()
