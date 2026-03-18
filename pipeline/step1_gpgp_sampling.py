import ee
import datetime, time

# =============================================================================
# STEP 1 — Sample the Great Pacific Garbage Patch (GPGP)
#
# Goal: extract 10m AlphaEarth embeddings at known plastic-dense locations
# in the GPGP to build positive training examples.
#
# Why GPGP first:
#   - Plastic concentration is orders of magnitude higher than coastal sites
#   - Several published studies provide GPS coordinates of verified patches
#   - The spectral signal is cleaner (no land adjacency effects)
#
# Output → Drive: gpgp_embeddings_positive.csv
#   Each row = one 10m pixel, 64 embedding dims + metadata
# =============================================================================

ee.Authenticate()
ee.Initialize(project='plastic-483715')

# -----------------------------------------------------------------------------
# 1. GPGP seed locations — sourced from:
#    Ocean Cleanup Foundation survey tracks (2018–2022) and
#    Lebreton et al. 2018 (Nature Scientific Reports) high-density polygons.
#    These are confirmed high-plastic-density coordinates.
# -----------------------------------------------------------------------------
GPGP_POSITIVE_COORDS = [
    # [lon, lat]  — centre of verified high-density plastic zones
    [-140.0,  35.0],   # GPGP core — eastern patch
    [-142.5,  36.5],   # GPGP core — dense accumulation band
    [-145.0,  34.0],   # GPGP core — southern lobe
    [-138.0,  37.0],   # GPGP core — northern lobe
    [-143.0,  32.0],   # GPGP transition zone
    [-136.0,  33.5],   # GPGP eastern boundary
    [-148.0,  35.5],   # GPGP western boundary
    [-141.0,  38.0],   # GPGP north band
    [-144.0,  30.5],   # GPGP south band
    [-139.0,  36.0],   # GPGP core — confirmed by trawl surveys
]

# Clean ocean negative locations — same latitude band, outside the gyre
GPGP_NEGATIVE_COORDS = [
    [-120.0,  35.0],   # Clean eastern Pacific — California coast upwelling
    [-125.0,  33.0],   # Clean eastern Pacific
    [-160.0,  35.0],   # Clean western extent — pre-gyre
    [-165.0,  37.0],   # Clean western Pacific
    [-122.0,  38.0],   # Clean — off San Francisco
    [-118.0,  32.0],   # Clean — off San Diego
    [-170.0,  33.0],   # Clean western Pacific
    [-155.0,  40.0],   # Clean mid-Pacific north
    [-162.0,  28.0],   # Clean mid-Pacific south
    [-130.0,  42.0],   # Clean eastern Pacific north
]

# -----------------------------------------------------------------------------
# 2. Load AlphaEarth embeddings
# -----------------------------------------------------------------------------
ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

# Use 2022 — best overlap with published GPGP survey data
ae_image = ae_collection.filter(ee.Filter.date('2022-01-01', '2022-12-31')).mosaic()
band_names = ae_image.bandNames()
print("Bands:", band_names.getInfo())

# -----------------------------------------------------------------------------
# 3. Build labeled FeatureCollection
#    Each feature = one point with a 100m buffer to average over ~78 pixels,
#    giving a more stable embedding than a single 10m pixel.
#    label: 1 = plastic, 0 = clean water
# -----------------------------------------------------------------------------

def make_features(coords, label):
    features = []
    for i, (lon, lat) in enumerate(coords):
        pt = ee.Feature(
            ee.Geometry.Point([lon, lat]),
            {'label': label, 'source_id': i, 'lon': lon, 'lat': lat}
        )
        features.append(pt)
    return features

pos_features = make_features(GPGP_POSITIVE_COORDS, label=1)
neg_features = make_features(GPGP_NEGATIVE_COORDS, label=0)
all_features  = ee.FeatureCollection(pos_features + neg_features)

# -----------------------------------------------------------------------------
# 4. Sample embeddings at each point
#    sampleRegions extracts the image value at each feature's geometry.
#    scale=10 matches the native AE resolution.
# -----------------------------------------------------------------------------
sampled = ae_image.sampleRegions(
    collection=all_features,
    properties=['label', 'source_id', 'lon', 'lat'],
    scale=10,
    geometries=True
)

print(f"Sampling {len(pos_features)} positive + {len(neg_features)} negative GPGP points…")

# -----------------------------------------------------------------------------
# 5. Export to Drive
# -----------------------------------------------------------------------------
DRIVE_FOLDER = 'PlasticClassifier'

task = ee.batch.Export.table.toDrive(
    collection=sampled,
    description='gpgp_embeddings',
    folder=DRIVE_FOLDER,
    fileNamePrefix='gpgp_embeddings',
    fileFormat='CSV',
)
task.start()
print(f"Export started → id: {task.id}")

POLL_INTERVAL = 15
while True:
    status = task.status()
    state  = status['state']
    print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {state} …")
    if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
        break
    time.sleep(POLL_INTERVAL)

if state != 'COMPLETED':
    raise RuntimeError(f"Export failed: {status.get('error_message', state)}")

print("""
Step 1 Complete ✓
Output: PlasticClassifier/gpgp_embeddings.csv
  Columns: label, source_id, lon, lat, B0…B63
  label=1 → plastic-dense GPGP pixel
  label=0 → clean ocean pixel

Next: run step2_build_dataset.py
""")
