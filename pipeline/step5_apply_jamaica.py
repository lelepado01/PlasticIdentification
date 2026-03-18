import ee
import datetime, time

# =============================================================================
# STEP 4 — Apply Classifier to Jamaica / Kingston Area
#
# Loads the trained RF classifier from the GEE Asset created in Step 3
# and applies it at full 10m resolution over the search area.
#
# Key difference from the original cosine-similarity approach:
#   - We're not comparing to one reference pixel anymore
#   - The RF has learned what plastic looks like across 3 environments
#     (beach, river, ocean) and can generalise to Kingston's specific context
#   - Output is P(plastic) per 10m pixel, not a binary mask
#
# Outputs → Drive: PlasticClassifier/
#   jamaica_plastic_probability.tif   — P(plastic) at 10m, water-masked
#   jamaica_rgb_overview.tif          — true-colour S2 for visual context
#   jamaica_detections.geojson        — patches where P > threshold
# =============================================================================

ee.Authenticate()
ee.Initialize(project='plastic-483715')

ASSET_ID     = 'projects/plastic-483715/assets/plastic_rf_classifier'
DRIVE_FOLDER = 'PlasticClassifier'
YEAR         = '2024'

# -----------------------------------------------------------------------------
# 1. Study area — Kingston harbour + 10km radius
# -----------------------------------------------------------------------------
ref_point   = ee.Geometry.Point([-76.817179, 17.974251])
search_area = ref_point.buffer(10000)

# -----------------------------------------------------------------------------
# 2. Reconstruct feature image (same as Step 3)
#    Must be identical band stack: B0…B63 + cosine_ref_kingston
# -----------------------------------------------------------------------------
ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
ae_2024 = ae_collection.filter(ee.Filter.date(f'{YEAR}-01-01', f'{YEAR}-12-31')).mosaic()
band_names = ae_2024.bandNames()

ref_dict = ae_2024.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=ref_point.buffer(50),
    scale=10,
    maxPixels=1e6
)
ref_image   = ee.Image.constant(ref_dict.values(band_names)).rename(band_names)
dot_product = ae_2024.multiply(ref_image).reduce(ee.Reducer.sum())
pixel_norm  = ae_2024.pow(2).reduce(ee.Reducer.sum()).sqrt()
ref_norm    = ref_image.pow(2).reduce(ee.Reducer.sum()).sqrt()
cosine_sim  = (dot_product.divide(pixel_norm).divide(ref_norm)
               .rename('cosine_ref_kingston'))

feature_image = ae_2024.addBands(cosine_sim)

# -----------------------------------------------------------------------------
# 3. Water + shoreline mask (JRC, 1km dilation) — same as before
#    Apply BEFORE classification to avoid wasting compute on land pixels
#    and to prevent inland false positives.
# -----------------------------------------------------------------------------
jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
water_mask_30m = jrc.select('occurrence').gt(0)
water_buffer   = (water_mask_30m
                  .focal_max(radius=1000, kernelType='circle', units='meters')
                  .reproject(crs='EPSG:4326', scale=30))

feature_image_masked = feature_image.updateMask(water_buffer)

# -----------------------------------------------------------------------------
# 4. Load classifier and apply
# -----------------------------------------------------------------------------
print(f"Loading classifier from {ASSET_ID}…")
classifier = ee.Classifier.load(ASSET_ID)

# classify() returns an image with one band named 'classification'
# In PROBABILITY mode this is P(plastic) ∈ [0.0, 1.0]
prob_image = (feature_image_masked
              .classify(classifier)
              .rename('plastic_probability')
              .clip(search_area))

print("Classifier applied.")

# -----------------------------------------------------------------------------
# 5. Threshold and vectorise detections
#
#    Two thresholds:
#      HIGH   P > 0.8 → report to field team immediately
#      MEDIUM P > 0.6 → flag for visual review
#
#    We export both as separate layers in the GeoJSON.
# -----------------------------------------------------------------------------
THRESHOLD_HIGH   = 0.8
THRESHOLD_MEDIUM = 0.6

def vectorise_threshold(prob_img, threshold, label):
    mask = prob_img.gt(threshold)
    return (mask
            .selfMask()
            .reduceToVectors(
                reducer=ee.Reducer.mean(),
                geometry=search_area,
                scale=10,             # 10m — native resolution
                maxPixels=1e10,
                geometryType='polygon',
                eightConnected=True,  # connect adjacent plastic pixels
                labelProperty='confidence'
            )
            .map(lambda f: f.set('confidence', label)
                            .set('threshold', threshold)))

detections_high   = vectorise_threshold(prob_image, THRESHOLD_HIGH,   'high')
detections_medium = vectorise_threshold(prob_image, THRESHOLD_MEDIUM, 'medium')

# Merge and add mean probability score per polygon
all_detections = detections_high.merge(
    # medium minus high — avoid duplicating high-confidence ones
    detections_medium.filter(
        ee.Filter.bounds(
            detections_high.geometry().buffer(1).complement()
        )
    )
)

# Attach mean probability to each detection polygon
detections_scored = prob_image.reduceRegions(
    collection=all_detections,
    reducer=ee.Reducer.mean().setOutputs(['mean_prob']),
    scale=10
)

# -----------------------------------------------------------------------------
# 6. S2 RGB for visual context (same as before, unmasked)
# -----------------------------------------------------------------------------
s2_rgb = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(search_area)
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(['B4', 'B3', 'B2'])
            .median()
            .clamp(0, 3000).divide(3000).multiply(255).toUint8()
            .rename(['red', 'green', 'blue']))

# -----------------------------------------------------------------------------
# 7. Export all three outputs
# -----------------------------------------------------------------------------

def start_export_image(image, name, scale=10):
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=search_area,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e10,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True},
    )
    task.start()
    print(f"  Started image export: {name}  id={task.id}")
    return task

def start_export_table(fc, name):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        fileFormat='GeoJSON',
        selectors=['confidence', 'threshold', 'mean_prob']
    )
    task.start()
    print(f"  Started table export: {name}  id={task.id}")
    return task

print("\nStarting exports…")
tasks = {
    'probability_raster': start_export_image(prob_image,        'jamaica_plastic_probability'),
    'rgb_overview':       start_export_image(s2_rgb.toFloat(),  'jamaica_rgb_overview'),
    'detections':         start_export_table(detections_scored, 'jamaica_detections'),
}

# -----------------------------------------------------------------------------
# 8. Poll all tasks
# -----------------------------------------------------------------------------
print("\nWaiting for exports…")
POLL_INTERVAL = 20
pending = dict(tasks)

while pending:
    time.sleep(POLL_INTERVAL)
    done_keys = []
    for name, task in pending.items():
        status = task.status()
        state  = status['state']
        print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {name}: {state}")
        if state == 'COMPLETED':
            print(f"  ✓ {name}")
            done_keys.append(name)
        elif state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(
                f"{name} failed: {status.get('error_message', state)}"
            )
    for k in done_keys:
        del pending[k]

print(f"""
Step 4 Complete ✓
Output files in PlasticClassifier/:

  jamaica_plastic_probability.tif
    → Float32 GeoTIFF, 10m, water-masked
    → Band 1: P(plastic) ∈ [0.0, 1.0]
    → Load: rasterio.open(...).read(1)

  jamaica_rgb_overview.tif
    → True-colour S2, 10m, full 10km area
    → Use as background in QGIS

  jamaica_detections.geojson
    → Polygon per detection cluster
    → Properties: confidence (high/medium), threshold, mean_prob
    → Drag over RGB in QGIS to review

Recommended next steps:
  1. Open RGB in QGIS, overlay detections.geojson
  2. Visually inspect high-confidence polygons first
  3. For any confirmed true positives, add to Step 2 dataset and retrain
  4. Increase SAMPLES_PER_REGION and add more positive sites to improve recall
""")
