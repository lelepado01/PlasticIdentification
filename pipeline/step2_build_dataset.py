import ee
import datetime, time

# =============================================================================
# STEP 2 — Build Full Labeled Dataset
#
# Extends Step 1 with:
#   - Beach plastic positives  (known polluted beaches)
#   - Beach clean negatives    (verified clean beaches)
#   - River mouth positives    (high plastic export rivers)
#   - River mouth negatives    (clean river mouths)
#   - GPGP embeddings already collected in Step 1
#
# Sampling strategy:
#   Rather than single points, we sample a GRID of 10m pixels within each
#   labeled region. This gives the classifier real pixel-level variance
#   instead of smoothed averages, which is what it will see at inference.
#
# Output → Drive: dataset_beach.csv, dataset_ocean.csv, dataset_river.csv
# =============================================================================

ee.Authenticate()
ee.Initialize(project='plastic-483715')

ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
ae_2022 = ae_collection.filter(ee.Filter.date('2022-01-01', '2022-12-31')).mosaic()
band_names = ae_2022.bandNames()

DRIVE_FOLDER = 'PlasticClassifier'
YEAR = '2022'

# =============================================================================
# LABELED REGIONS
# Each entry: (lon, lat, buffer_m, label, environment, name)
#   buffer_m: radius around point to grid-sample from
#   label   : 1=plastic, 0=clean
#   environment: 'beach' | 'ocean' | 'river'
# =============================================================================

LABELED_REGIONS = [

    # ── BEACH POSITIVES ──────────────────────────────────────────────────────
    # Sources: Ocean Conservancy ICC data, published field surveys
    (-76.817179, 17.974251, 200, 1, 'beach', 'kingston_interceptor'),      # Jamaica — reference site
    (106.8456,  -6.1000,   300, 1, 'beach', 'jakarta_bay'),                # Indonesia — highly polluted
    (120.9842,  14.5995,   300, 1, 'beach', 'manila_bay'),                 # Philippines
    (3.3792,    6.4550,    200, 1, 'beach', 'lagos_bar_beach'),             # Nigeria
    (91.8325,   21.4272,   200, 1, 'beach', 'cox_bazar'),                  # Bangladesh
    (80.2707,   13.0827,   200, 1, 'beach', 'chennai_marina'),             # India

    # ── BEACH NEGATIVES ──────────────────────────────────────────────────────
    (-149.8989, -17.5500,  200, 0, 'beach', 'bora_bora'),                  # French Polynesia — pristine
    (-109.4500, -27.1127,  200, 0, 'beach', 'easter_island'),              # Easter Island — remote
    (-176.6413, -44.3500,  200, 0, 'beach', 'antipodes_island'),           # Antipodes — uninhabited
    (-63.0610,  18.0731,   200, 0, 'beach', 'anguilla_shoal_bay'),         # Caribbean — clean tourist
    (115.1889, -33.8688,   200, 0, 'beach', 'perth_cottesloe'),            # Australia — monitored clean
    (-9.4200,   38.6800,   200, 0, 'beach', 'setubal_portugal'),           # Portugal — EU Blue Flag

    # ── RIVER MOUTH POSITIVES ────────────────────────────────────────────────
    # Top plastic-exporting rivers (Lebreton et al. 2017, Schmidt et al. 2017)
    (106.7500,  -6.0800,   400, 1, 'river', 'citarum_mouth'),              # Indonesia — #1 plastic river
    (120.6331,  15.1430,   300, 1, 'river', 'pampanga_mouth'),             # Philippines
    (91.9000,   22.3500,   300, 1, 'river', 'buriganga_mouth'),            # Bangladesh
    (103.8800,   1.2800,   300, 1, 'river', 'singapore_kallang'),          # Singapore
    (-76.8500,  17.9800,   300, 1, 'river', 'kingston_harbour_river'),     # Jamaica — Hope River
    (28.9500,   41.0200,   300, 1, 'river', 'bosphorus_south'),            # Turkey

    # ── RIVER MOUTH NEGATIVES ────────────────────────────────────────────────
    (-123.1200,  49.3000,  300, 0, 'river', 'fraser_mouth_bc'),            # Canada — clean
    (174.7633, -36.8485,   300, 0, 'river', 'auckland_waitemata'),         # NZ — clean harbour
    (-68.3000,  -54.8000,  300, 0, 'river', 'beagle_channel'),             # Patagonia — pristine
    (18.9553,   69.6489,   300, 0, 'river', 'tromso_fjord'),               # Norway — clean
    (-122.4000,  37.8000,  300, 0, 'river', 'san_francisco_bay_north'),    # USA — monitored
    (151.2093, -33.8688,   300, 0, 'river', 'sydney_manly_cove'),          # Australia — clean

    # ── OCEAN POSITIVES (GPGP supplement) ───────────────────────────────────
    # Additional confirmed accumulation zones beyond Step 1
    (-28.5000,  26.0000,   500, 1, 'ocean', 'north_atlantic_gyre'),        # NAG centre
    (175.0000,  30.0000,   500, 1, 'ocean', 'north_pacific_west'),         # Western patch
    (-29.0000, -30.0000,   500, 1, 'ocean', 'south_atlantic_gyre'),        # SAG
    (80.0000,  -30.0000,   500, 1, 'ocean', 'indian_ocean_gyre'),          # IOG

    # ── OCEAN NEGATIVES ──────────────────────────────────────────────────────
    (-90.0000,   0.0000,   500, 0, 'ocean', 'equatorial_pacific_clean'),
    (160.0000, -10.0000,   500, 0, 'ocean', 'south_pacific_clean'),
    (-40.0000, -50.0000,   500, 0, 'ocean', 'south_atlantic_clean'),
    (40.0000,  -40.0000,   500, 0, 'ocean', 'south_indian_clean'),
]

# =============================================================================
# Build FeatureCollection and sample a GRID within each region
#
# stratifiedSample draws N random 10m pixels from within each buffer polygon,
# preserving the label property. This gives real pixel-level variance.
# =============================================================================

SAMPLES_PER_REGION = 200   # 10m pixels per labeled region

def sample_region(lon, lat, buffer_m, label, environment, name):
    """Return a sampled FC from one labeled region."""
    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    region_img = ae_2022.clip(geom)

    samples = region_img.sample(
        region=geom,
        scale=10,
        numPixels=SAMPLES_PER_REGION,
        seed=42,
        geometries=True,
        dropNulls=True
    )
    # Tag every pixel with its region metadata
    return samples.map(lambda f: f.set({
        'label':       label,
        'environment': environment,
        'site_name':   name,
        'lon':         lon,
        'lat':         lat,
    }))

print(f"Sampling {len(LABELED_REGIONS)} regions × {SAMPLES_PER_REGION} pixels…")

sampled_list = [
    sample_region(lon, lat, buf, lbl, env, name)
    for lon, lat, buf, lbl, env, name in LABELED_REGIONS
]

full_dataset = ee.FeatureCollection(sampled_list).flatten()

# =============================================================================
# Export one CSV per environment type — easier to inspect and balance
# =============================================================================

def export_and_poll(fc, name, description):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        fileFormat='CSV',
    )
    task.start()
    print(f"  Started: {name}  id={task.id}")
    return task

tasks = {}
for env in ['beach', 'river', 'ocean']:
    subset = full_dataset.filter(ee.Filter.eq('environment', env))
    tasks[env] = export_and_poll(
        subset,
        f'dataset_{env}',
        f'dataset_{env}'
    )

print("\nPolling exports…")
POLL_INTERVAL = 20
pending = dict(tasks)

while pending:
    time.sleep(POLL_INTERVAL)
    done = []
    for env, task in pending.items():
        state = task.status()['state']
        print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {env}: {state}")
        if state == 'COMPLETED':
            print(f"  ✓ {env} done")
            done.append(env)
        elif state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(
                f"{env} export failed: {task.status().get('error_message', state)}"
            )
    for env in done:
        del pending[env]

print("""
Step 2 Complete ✓
Output files in PlasticClassifier/:
  dataset_beach.csv   — beach pixels, label 0/1, 64 embedding dims
  dataset_river.csv   — river mouth pixels
  dataset_ocean.csv   — open ocean pixels

Each row is one 10m pixel. Columns:
  label, environment, site_name, lon, lat, B0 … B63

Class balance check (run before step 3):
  import pandas as pd
  df = pd.concat([pd.read_csv(f) for f in ['dataset_beach.csv','dataset_river.csv','dataset_ocean.csv']])
  print(df.groupby(['environment','label']).size())

Next: run step3_train_classifier.py
""")
