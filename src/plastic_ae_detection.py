import ee

# =============================================================================
# Plastic Patch Detection via AlphaEarth Spectral Embeddings
# Strategy: Cosine similarity at 10m → aggregate to 1km patches
# =============================================================================

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='plastic-483715')

# 2. Reference point: Kingston Interceptor Area 2 (known high-plastic site)
ref_point = ee.Geometry.Point([-76.817179, 17.974251])
search_area = ref_point.buffer(5000)  # 50 km search radius

# 3. Load AlphaEarth 2024 Annual Embeddings (64-band, 10m resolution)
ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
ae_2024 = ae_collection.filter(ee.Filter.date('2024-01-01', '2024-12-31')).mosaic()

band_names = ae_2024.bandNames()
print("Embedding bands:", band_names.getInfo())

# 4. Extract reference embedding
#    Mean over a 50m buffer at the known plastic site.
#    NOTE: We use the band-ordered .values() to guarantee dimension alignment
#    with the image bands — plain dict ordering is not reliable.
ref_dict = ae_2024.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=ref_point.buffer(50),
    scale=10,
    maxPixels=1e6
)

# Build a constant image from the reference vector, bands in the same order
# as ae_2024. This is the key step that the original script got wrong —
# ee.Array has no guaranteed band order, making dot products meaningless.
ref_image = ee.Image.constant(ref_dict.values(band_names)).rename(band_names)

# 5. Cosine similarity at native 10m resolution
#    cos(θ) = (A · B) / (|A| * |B|)
#    Range: -1 (opposite) to +1 (identical spectral signature)
dot_product   = ae_2024.multiply(ref_image).reduce(ee.Reducer.sum())
pixel_norm    = ae_2024.pow(2).reduce(ee.Reducer.sum()).sqrt()
ref_norm      = ref_image.pow(2).reduce(ee.Reducer.sum()).sqrt()

cosine_sim_10m = (dot_product
                  .divide(pixel_norm)
                  .divide(ref_norm)
                  .rename('cosine_similarity'))

# 6. Aggregate to 1 km² patches
#    Arithmetic ops (multiply/divide) strip the source projection from the
#    output image, leaving it "undefined". reduceResolution() needs a valid
#    native projection so it knows the input pixel grid to aggregate from.
#    Fix: re-attach the AlphaEarth native projection (EPSG:4326 at 10m) with
#    setDefaultProjection() BEFORE calling reduceResolution().
#
#    Using mean: "What fraction of this km² looks like plastic?"
#    Swap to ee.Reducer.max() to flag km² cells containing *any*
#    highly plastic-like pixels (more sensitive, more false positives).
# AlphaEarth embeddings are natively EPSG:4326 at 10m.
# DO NOT use ae_2024.projection() — mosaic() discards the source projection,
# so the returned object is undefined and silently breaks reduceResolution
# in the export backend. Hardcode the known spec instead.
native_proj = ee.Projection('EPSG:4326').atScale(10)

cosine_sim_1km = (cosine_sim_10m
                  .setDefaultProjection(native_proj)
                  .reduceResolution(
                      reducer=ee.Reducer.mean(),
                      maxPixels=10000
                  )
                  .reproject(crs='EPSG:4326', scale=1000)
                  .rename('cosine_similarity'))

# 7. Build water + shoreline mask FIRST using JRC Global Surface Water v1.4
#
#    occurrence band: 0 = never water, 100 = always water (Landsat 1984–2021)
#    We take occurrence > 0 (ever observed as water) then dilate 1km to include:
#      - beaches and dry sand above the tide line
#      - river mouths and tidal flats (intermittently wet)
#      - nearshore zones where floating plastic accumulates
#
#    The dilation is done at 30m then snapped to the 1km cosine grid so
#    every 1km cell that contains ANY water-adjacent 30m pixel is included.
jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')

water_mask_30m = jrc.select('occurrence').gt(0)

# Dilate 1km at native 30m resolution, then snap to 1km grid
water_buffer_1km = (water_mask_30m
                    .focal_max(radius=1000, kernelType='circle', units='meters')
                    .reproject(crs='EPSG:4326', scale=30)       # compute at 30m
                    .reproject(crs='EPSG:4326', scale=1000))    # snap to 1km grid

# 7b. Mask cosine similarity to water-adjacent cells BEFORE thresholding.
#
#    This is the critical fix: if you threshold over all pixels first, land
#    covers (forest, urban, bare soil) dominate the top percentiles because
#    they have stronger/cleaner spectral signatures than messy water pixels.
#    By masking first, the percentile is computed exclusively among
#    coastal/aquatic candidates — so "top 25%" means top among those cells.
cosine_sim_water = cosine_sim_1km.updateMask(water_buffer_1km)

# Top 25% of water-adjacent cells by cosine similarity.
# Using 75th percentile (not 95th) because after land exclusion the pool
# is much smaller (~10–30 cells in a 10km radius) and we want enough
# candidates to inspect visually — tighten to 90th once you've validated.
stats = cosine_sim_water.reduceRegion(
    reducer=ee.Reducer.percentile([75]),
    geometry=search_area,
    scale=1000,
    maxPixels=1e9
)
threshold = stats.getNumber('cosine_similarity')

high_likelihood_mask = cosine_sim_water.gt(threshold)

# 8. Convert to vectors so you can get actual patch coordinates
#    for downstream hi-res image retrieval
high_likelihood_patches = (high_likelihood_mask
                            .selfMask()
                            .reduceToVectors(
                                reducer=ee.Reducer.countEvery(),
                                geometry=search_area,
                                scale=1000,
                                maxPixels=1e9,
                                geometryType='centroid',   # centroid per 1km cell
                                eightConnected=False,
                                labelProperty='patch_id'
                            ))

# 9. Build the two export layers, both masked to flagged 1km cells only
#
#    Layer A — Sentinel-2 SR (10m, 6 bands): the actual optical signal you
#              need for visual inspection and pixel-level plastic classifiers.
#    Layer B — Cosine similarity at 10m: the raw spectral distance surface
#              that drove the flagging, useful as an additional feature.
#
#    Both are masked so only pixels inside high-likelihood 1km cells are
#    exported — nothing outside the flagged areas is written to Drive.

import datetime, time

DRIVE_FOLDER = 'EarthEngineExports/PlasticPatches'
YEAR         = '2024'

# --- 9a. Sentinel-2 SR (cloud-filtered median composite, 10m) ----------------
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(search_area)
        .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])  # Blue,Green,Red,NIR,SWIR1,SWIR2
        .median()
        .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']))

# Upscale the 1km binary mask back to 10m so it aligns with S2 pixels
mask_10m = (high_likelihood_mask
              .reproject(crs='EPSG:4326', scale=10))

s2_masked  = s2.updateMask(mask_10m).clip(search_area)

# --- 9b. Cosine similarity at 10m, masked ------------------------------------
cosine_masked = cosine_sim_10m.updateMask(mask_10m).clip(search_area)

# --- 9c. Stack both into one export image ------------------------------------
#    Single GeoTIFF with 7 bands:
#      1-6 → S2 reflectance (blue…swir2, scale 0–10000)
#      7   → cosine_similarity (float32, range -1…+1)
export_image = (s2_masked
                  .addBands(cosine_masked.rename('cosine_similarity'))
                  .toFloat())

# =============================================================================
# 10. Export to Drive and poll
# =============================================================================

def poll(task, label):
    """Block until task completes; raise on failure."""
    POLL_INTERVAL = 20
    while True:
        status = task.status()
        state  = status['state']
        print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {label}: {state} …")
        if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
            break
        time.sleep(POLL_INTERVAL)
    if state != 'COMPLETED':
        raise RuntimeError(
            f"{label} export failed: {status.get('error_message', state)}"
        )
    print(f"  ✓ {label} complete")

# --- 10a. RGB overview of the full 10km search area (unmasked) --------------
#    True-colour S2 composite (R=B4, G=B3, B=B2) stretched to 8-bit.
#    This is your visual context — open it in QGIS/Preview alongside the
#    masked patch GeoTIFF to see exactly what the flagged areas look like.
#    No mask applied; you see land, water, cloud gaps, and plastic candidates.
s2_rgb = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(search_area)
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(['B4', 'B3', 'B2'])          # Red, Green, Blue
            .median()
            # S2 SR surface reflectance is 0–10000; rescale to 0–255 for display
            # Stretch: 0–3000 → 0–255 (typical for coastal/water scenes)
            .clamp(0, 3000)
            .divide(3000)
            .multiply(255)
            .toUint8()
            .rename(['red', 'green', 'blue']))

task_rgb = ee.batch.Export.image.toDrive(
    image=s2_rgb,
    description='overview_rgb_10km',
    folder=DRIVE_FOLDER,
    fileNamePrefix='overview_rgb_10km',
    region=search_area,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e10,
    fileFormat='GeoTIFF',
    formatOptions={'cloudOptimized': True},
)
task_rgb.start()
print(f"RGB overview export started  →  id: {task_rgb.id}")

# --- 10c. Image export (GeoTIFF) ---------------------------------------------
task_img = ee.batch.Export.image.toDrive(
    image=export_image,
    description='plastic_patches_s2_cosine',
    folder=DRIVE_FOLDER,
    fileNamePrefix='plastic_patches_s2_cosine',
    region=search_area,
    scale=10,                        # native S2 resolution
    crs='EPSG:4326',
    maxPixels=1e10,
    fileFormat='GeoTIFF',
    formatOptions={'cloudOptimized': True},  # COG → readable by rasterio/QGIS
)
task_img.start()
print(f"Image export started  →  id: {task_img.id}")

# --- 10d. Patch footprints as GeoJSON (for indexing which tiles to load) -----
#    We still export the vector metadata — it's lightweight and lets you
#    index into the GeoTIFF by patch bounding box without loading the whole file.
high_likelihood_patches = cosine_sim_1km.reduceRegions(
    collection=(high_likelihood_mask
                .selfMask()
                .reduceToVectors(
                    reducer=ee.Reducer.countEvery(),
                    geometry=search_area,
                    scale=1000,
                    maxPixels=1e9,
                    geometryType='polygon',   # full 1km square, not just centroid
                    eightConnected=False,
                    labelProperty='patch_id'
                )),
    reducer=ee.Reducer.mean().setOutputs(['similarity']),
    scale=1000
)

task_vec = ee.batch.Export.table.toDrive(
    collection=high_likelihood_patches,
    description='plastic_patches_index',
    folder=DRIVE_FOLDER,
    fileNamePrefix='plastic_patches_index',
    fileFormat='GeoJSON',
    selectors=['patch_id', 'similarity']
)
task_vec.start()
print(f"Index export started  →  id: {task_vec.id}")

# --- 10e. Poll all three tasks in parallel ----------------------------------------
print("\nWaiting for exports…")
poll(task_rgb, 'RGB overview GeoTIFF')
poll(task_img, 'S2+Cosine GeoTIFF')
poll(task_vec, 'Patch index GeoJSON')

print(f"""
Phase 1 Complete ✓
Drive folder : {DRIVE_FOLDER}
  overview_rgb_10km.tif            ← true-colour RGB, full 10km area, 10m, 8-bit
  plastic_patches_s2_cosine.tif   ← 7-band GeoTIFF (S2 + cosine), 10m, COG
  plastic_patches_index.geojson   ← patch footprints + similarity scores

Load in rasterio:
  import rasterio
  with rasterio.open("plastic_patches_s2_cosine.tif") as src:
      data = src.read()   # shape: (7, H, W)
      # bands 0-5 → S2 reflectance (/10000 to get 0-1)
      # band  6   → cosine similarity
""")