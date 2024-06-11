# KDDCUP 2024 OAG-Challenge PST solution
## Authors

- NTTDOCMOLABZ
  - masatoh 
  - K.O. 
  - yukko08

## Solution Overview
figure goes here

## Requires
- Linux (Amazon linux 2 was used in our team)
- CUDA 12.1.0

## Environment setup
```
$ docker compose up -d
$ docker exec -it kdd2024-pst /bin/bash
$ <Inside-container> poetry install
```

## Dataset
- Provided PST dataset available at the [competion page](https://www.biendata.xyz/competition/pst_kdd_2024/data/)
  - https://www.dropbox.com/scl/fi/namx1n55xzqil4zbkd5sv/PST.zip?rlkey=impcbm2acqmqhurv2oj0xxysx&dl=1
  - https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/PST/PST-test-public.zip
- DBLP dataset
  - DBLP-Citation-network V15 available at [here](https://open.aminer.cn/open/article?id=655db2202ab17a072284bc0c).
  
- OAG dataset 
  - OAG V3.1 dataset Publication data available at [here](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf)
     - publication_1.zip ~ publication_14.zip

## Directory structure
Please put the dataset files shown below.

```
```


## Pre-process 
- 0-1 process_kddcup_data.py 
  - Extracts data from the DBLP dataset.


- 0-2 xml_parser.py (0-1)
  - Parses XML files from the provided train, validation, and test datasets as well as the DBLP dataset to extract titles, abstracts, keywords, organizations, venues, context, etc.

- 0-3 oagbert_title_gen.py (0-2)
  - Generates paper titles for missing values in the test dataset for the final submission.

- 0-4 dblp_feature.py (0-1)
  - Creates paper metadata features using the DBLP dataset and XML files. This script is based on the public baseline code available at https://github.com/THUDM/paper-source-trace/blob/main/rf/process_data.py
  

- 0-5 dblp_feature_2.py (0-1)
  - Modifies dblp_feature.py  

- 0-6 sciber_encode_3.py (0-3)
  - Performs SciBERT encoding for each sentence of the title, abstract, keywords, organization, venue, and context, which are outputs from xml_parser.py. It also calculates cosine similarities between the target paper and the source paper using each feature.  

- 0-7 nb/feature_post_process.ipynb (0-3)
  - Adjusts cosine similarities.

- 0-8 feature_generation/Fill_paper_info_by_OAG.ipynb (0-3)
  - Refer to feature_generation/README.md for details.  

- 0-9 feature_generation/OAGBERT_embedding_cossim.ipynb (0-3)
  - Refer to feature_generation/README.md for details.   

- 0-10 oag_bert.py (0-3)
  - Encodes papers using the OAGBERT model and calculates the cosine similarity between the target and source papers.
  
## First Stage
- 1-1 cross_encoder.py (0-3)
  - Executes the cross-encoder model using the sentence-transformers library. We use the SciBERT model. The command to run this is `poetry run python cross_encoder.py --ubm --output_dir ce/default --train --prediction`.
  - Implements 5-fold cross-validation. GroupKFold.

- run_ce.sh
  - The entry point for cross_encoder.py.
  
## Second stage
- 2-1 nb/catboost_4.ipynb
  - Feature 0601 (baseline) (0-3/0-4/0-6/0-8/1-1)

- 2-2 nb/catboost_5_feature_wihout_emb.ipynb
  - Feature feat_without_emb (0-3/0-4/0-8/0-10/1-1)

- 2-3 nb/catboost6_with_oag_clean.ipynb
  - Feature oag_clean (0-3/0-5/0-6/0-7/0-8/0-9/1-1)

- 2-4 classifier.py (2-1/2-2)
  - Excecutes binary classification models. Catboost / LightGBM / RandomForest / SVM.
  - Implements 5-fold cross-validation. GroupKFold.

- 2-5 classifier_weight.py (classifier v2) (2-3)
  - Modified version of classifier.py.  
- run_classifier.sh
  - The entry point for classifier.py and classifier_weight.py.

## Ensemble
 - Ensemble
   - nb/ensemble_lb0413_oag_clean.ipynb
     - Catboost / LightGBM / RandomForest (2-1/2-4)
     - Support Vector Machine (2-2/2-4)
     - Catboost / LightGBM (2-5)
   - Ensemble method: average