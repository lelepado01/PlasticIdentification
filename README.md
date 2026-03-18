# Plastic Detection Pipeline — AlphaEarth Embeddings 

## Overview

For Val: 

```bash
python pipeline/download_swaths.py
# downloads the patches, 
# I'm not sure I can add you to the same google earth project
```

```
step1_gpgp_sampling.py      # verify embeddings work on known GPGP plastic
step2_build_dataset.py      # build labeled dataset (beach + river + ocean)
step3_train_classifier.py   # train RF, validate, save to GEE Asset
step4_apply_jamaica.py      # apply to Kingston area, export results
```


## Setup

```bash
pip install earthengine-api
ee-authenticate   # or run ee.Authenticate() in each script
```

GEE project: `plastic-483715`
GEE Asset path: `projects/plastic-483715/assets/plastic_rf_classifier`

---

## Running the pipeline

```bash
python pipeline/step1_gpgp_sampling.py
python pipeline/step2_build_dataset.py
python pipeline/step3_download_patches.py
```
