import ee

# =============================================================================
# Plastic Patch Detection via AlphaEarth Spectral Embeddings
# Strategy: Cosine similarity at 10m → aggregate to 1km patches
# =============================================================================

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='plastic-483715')

# 2. Reference point: Kingston Interceptor Area 2 (known high-plastic site)
ref_point = ee.Geometry.Point([-76.817179, 17.974251]) #17.957921, -76.819207
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
                  .rename('cosine_similarity_p95'))

# 7. Threshold: top 5% most similar 1km patches within search area
#    NOTE: After reduceRegion + percentile([95]), the key is '{band}_p95'
#    The original used percentile([19]) — the 19th percentile — which is
#    the bottom 81%, the opposite of what was intended.
stats = cosine_sim_1km.reduceRegion(
    reducer=ee.Reducer.percentile([95]),
    geometry=search_area,
    scale=1000,
    maxPixels=1e9
)
threshold = stats.getNumber('cosine_similarity_p95')

high_likelihood_mask = cosine_sim_1km.gt(threshold)

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

# 9. Attach cosine similarity score to each patch centroid
high_likelihood_patches = cosine_sim_1km.reduceRegions(
    collection=high_likelihood_patches,
    reducer=ee.Reducer.mean().setOutputs(['similarity']),
    scale=1000
)

# 10. Export server-side to Google Drive then download locally
#
#     WHY NOT getInfo()?
#     getInfo() re-materialises the full lazy computation graph
#     (including reduceResolution) in a single synchronous RPC call.
#     GEE rejects this because reduceResolution loses its projection
#     context when forced through that path. The correct pattern is
#     always Export → poll → download.
import io, json, pathlib, datetime, time

EXPORT_NAME = 'plastic_patches_kingston'
OUT_PATH    = pathlib.Path(f'{EXPORT_NAME}.geojson')

# --- 10a. Start async Drive export -------------------------------------------
task = ee.batch.Export.table.toDrive(
    collection=high_likelihood_patches,
    description=EXPORT_NAME,
    folder='EarthEngineExports',       # Drive folder (created automatically)
    fileNamePrefix=EXPORT_NAME,
    fileFormat='GeoJSON',
    selectors=['similarity', 'patch_id']
)
task.start()
print(f"Export task started  →  id: {task.id}")

# --- 10b. Poll until complete -------------------------------------------------
POLL_INTERVAL = 15   # seconds
while True:
    status = task.status()
    state  = status['state']
    print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {state} …")
    if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
        break
    time.sleep(POLL_INTERVAL)

if state != 'COMPLETED':
    raise RuntimeError(f"Export did not complete: {status.get('error_message', state)}")
print("Export complete. Downloading from Drive…")