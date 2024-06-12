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

import util as utils
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s"
)  # include timestamp


random.seed(1)
DATA_TRACE_DIR = "../dataset"


def extract_paper_info_from_dblp():
    data_dir = join(DATA_TRACE_DIR, "PST")
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    # papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers_test_pub = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    paper_dict_open = {}
    dblp_fname = "../dataset/DBLP-Citation-network-V15.json"
    # dblp_fname = "dblp_v14.json"
    with open(dblp_fname, "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            if i % 10000 == 0:
                logger.info("reading papers %d", i)
            # print(line.strip().rstrip(","))
            try:
                paper_tmp = json.loads(line.strip().rstrip(",").rstrip("]"))
            except:
                print(line.strip().rstrip(",").rstrip("]"))
                raise ValueError()

            paper_dict_open[paper_tmp["id"]] = paper_tmp

    paper_dict_hit = dd(dict)
    for paper in tqdm(papers_train + papers_test_pub):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = [cur_pid] + ref_ids
        for pid in pids:
            if pid not in paper_dict_open:
                continue
            cur_paper_info = paper_dict_open[pid]
            cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
            n_citation = cur_paper_info.get("n_citation", 0)
            title = cur_paper_info.get("title", "")
            year = cur_paper_info.get("year", -1)
            doc_type = cur_paper_info.get("doc_type", "other")
            venue = cur_paper_info.get("venue", "")
            urls = [a for a in cur_paper_info.get("url", [])]
            abstract = cur_paper_info.get("abstract", "")
            keywords = [a for a in cur_paper_info.get("keywords", [])]
            authors = [a for a in cur_paper_info.get("authors", {})]
            dois = cur_paper_info.get("doi", "")

            # abst = cur_paper_info.get("abstract", "")

            paper_dict_hit[pid] = {
                "authors": cur_authors,
                "n_citation": n_citation,
                "title": title,
                "year": year,
                "doc_type": doc_type,
                "venue": venue,
                "urls": urls,
                "abstract": abstract,
                "keywords": keywords,
                "authors": authors,
                "doi": dois,
            }

    print("number of papers after filtering", len(paper_dict_hit))
    os.makedirs("../data", exist_ok=True)
    utils.dump_json(paper_dict_hit, "../data", "paper_info_hit_from_dblp_test_pub.json")


if __name__ == "__main__":
    extract_paper_info_from_dblp()
