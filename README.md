# Plastic Detection Pipeline — AlphaEarth Embeddings + Random Forest

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

---

## Outputs

After Step 4, your Drive folder `PlasticClassifier/` will contain:

| File | Description |
|------|-------------|
| `jamaica_plastic_probability.tif` | Float32 GeoTIFF, P(plastic) 0–1, 10m |
| `jamaica_rgb_overview.tif` | True-colour S2, 10m, full area |
| `jamaica_detections.geojson` | Detection polygons with confidence scores |
| `gpgp_embeddings.csv` | Step 1 raw embeddings (for inspection) |
| `dataset_beach/river/ocean.csv` | Step 2 labeled training data |

---

## Tuning

**Too many false positives?**
- Raise `THRESHOLD_HIGH` in step4 to 0.85–0.90
- Add more clean-water negatives in the Kingston area to step2

**Missing known plastic sites?**
- Lower `THRESHOLD_MEDIUM` to 0.5
- Add more positive training samples from similar coastal environments
- Increase `SAMPLES_PER_REGION` in step2/step3

**Iterative improvement loop:**
1. Run step4, inspect detections in QGIS
2. For confirmed true positives: add coords to step2 LABELED_REGIONS (label=1)
3. For confirmed false positives: add coords as negatives (label=0)
4. Re-run step3 and step4 — accuracy improves quickly with local ground truth

---

## Data sources

- **AlphaEarth embeddings**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- **Sentinel-2 SR**: `COPERNICUS/S2_SR_HARMONIZED`
- **JRC Global Surface Water**: `JRC/GSW1_4/GlobalSurfaceWater`
- **GPGP coordinates**: Lebreton et al. 2018, Ocean Cleanup Foundation surveys
- **Polluted river mouths**: Schmidt et al. 2017, Lebreton et al. 2017
- **Polluted beaches**: Ocean Conservancy ICC dataset
