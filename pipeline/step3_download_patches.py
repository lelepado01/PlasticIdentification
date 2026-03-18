import ee
import io, time, datetime, pathlib
import numpy as np

# =============================================================================
# PATCH DOWNLOADER
#
# Replaces steps 3 & 4. Exports labeled 10m embedding patches to Drive as
# .npy files so you can train locally.
#
# GEE cannot write .npy directly, so the pipeline is:
#   GEE sample → CSV export to Drive → download here → convert → reupload .npy
#
# Output files in Drive: PlasticClassifier/npy/
#   X_train.npy   shape (N, 64)   float32 — AE embedding per 10m pixel
#   y_train.npy   shape (N,)      int8    — 0=clean, 1=plastic
#   meta.npy      shape (N,)      object  — site_name, environment per pixel
#
# Run after step2_build_dataset.py has completed.
# =============================================================================

ee.Authenticate()
ee.Initialize(project='plastic-483715')

DRIVE_FOLDER  = 'PlasticClassifier'
NPY_SUBFOLDER = 'PlasticClassifier/npy'
YEAR          = '2022'

# ── Same labeled regions as step 2 ──────────────────────────────────────────
LABELED_REGIONS = [
    # (lon, lat, buffer_m, label, environment, site_name)
    # BEACH +
    (-76.817179, 17.974251, 200, 1, 'beach', 'kingston_interceptor'),
    (106.8456,   -6.1000,  300, 1, 'beach', 'jakarta_bay'),
    (120.9842,   14.5995,  300, 1, 'beach', 'manila_bay'),
    (3.3792,      6.4550,  200, 1, 'beach', 'lagos_bar_beach'),
    (91.8325,    21.4272,  200, 1, 'beach', 'cox_bazar'),
    (80.2707,    13.0827,  200, 1, 'beach', 'chennai_marina'),
    # BEACH -
    (-149.8989, -17.5500,  200, 0, 'beach', 'bora_bora'),
    (-109.4500, -27.1127,  200, 0, 'beach', 'easter_island'),
    (-63.0610,   18.0731,  200, 0, 'beach', 'anguilla_shoal_bay'),
    (115.1889,  -33.8688,  200, 0, 'beach', 'perth_cottesloe'),
    (-9.4200,    38.6800,  200, 0, 'beach', 'setubal_portugal'),
    # RIVER +
    (106.7500,   -6.0800,  400, 1, 'river', 'citarum_mouth'),
    (120.6331,   15.1430,  300, 1, 'river', 'pampanga_mouth'),
    (91.9000,    22.3500,  300, 1, 'river', 'buriganga_mouth'),
    (-76.8500,   17.9800,  300, 1, 'river', 'kingston_hope_river'),
    (28.9500,    41.0200,  300, 1, 'river', 'bosphorus_south'),
    # RIVER -
    (-123.1200,  49.3000,  300, 0, 'river', 'fraser_mouth_bc'),
    (174.7633,  -36.8485,  300, 0, 'river', 'auckland_waitemata'),
    (-68.3000,  -54.8000,  300, 0, 'river', 'beagle_channel'),
    (18.9553,    69.6489,  300, 0, 'river', 'tromso_fjord'),
    (-122.4000,  37.8000,  300, 0, 'river', 'san_francisco_bay'),
    # OCEAN + (GPGP)
    (-140.0,     35.0,     500, 1, 'ocean', 'gpgp_core_east'),
    (-142.5,     36.5,     500, 1, 'ocean', 'gpgp_core_dense'),
    (-145.0,     34.0,     500, 1, 'ocean', 'gpgp_south_lobe'),
    (-138.0,     37.0,     500, 1, 'ocean', 'gpgp_north_lobe'),
    (-28.5000,   26.0000,  500, 1, 'ocean', 'north_atlantic_gyre'),
    (175.0000,   30.0000,  500, 1, 'ocean', 'north_pacific_west'),
    # OCEAN -
    (-120.0,     35.0,     500, 0, 'ocean', 'clean_eastern_pacific'),
    (-165.0,     37.0,     500, 0, 'ocean', 'clean_western_pacific'),
    (-90.0,       0.0,     500, 0, 'ocean', 'clean_equatorial_pacific'),
    (160.0,     -10.0,     500, 0, 'ocean', 'clean_south_pacific'),
    (-40.0,     -50.0,     500, 0, 'ocean', 'clean_south_atlantic'),
    (40.0,      -40.0,     500, 0, 'ocean', 'clean_south_indian'),
]

SAMPLES_PER_REGION = 100  # 10m pixels per site

# =============================================================================
# PHASE A — GEE: sample embeddings and export per-environment CSVs to Drive
# =============================================================================

ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
ae_image = ae_collection.filter(ee.Filter.date(f'{YEAR}-01-01', f'{YEAR}-12-31')).mosaic()
band_names = ae_image.bandNames()
print("Embedding bands:", len(band_names.getInfo()))

def sample_region(lon, lat, buffer_m, label, environment, site_name):
    geom    = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    samples = ae_image.sample(
        region=geom,
        scale=10,
        numPixels=SAMPLES_PER_REGION,
        seed=42,
        geometries=False,
        dropNulls=True
    )
    return samples.map(lambda f: f.set({
        'label':       label,
        'environment': environment,
        'site_name':   site_name,
        'site_lon':    lon,
        'site_lat':    lat,
    }))

print(f"\nSampling {len(LABELED_REGIONS)} regions at 10m…")
all_samples = ee.FeatureCollection(
    [sample_region(*r) for r in LABELED_REGIONS]
).flatten()

# Export one CSV per environment — keeps files manageable and lets you
# load subsets (e.g. beach-only) without parsing the full dataset.
export_tasks = {}
for env in ['beach', 'river']: # ocean
    subset = all_samples.filter(ee.Filter.eq('environment', env))
    task = ee.batch.Export.table.toDrive(
        collection=subset,
        description=f'patches_{env}',
        folder=DRIVE_FOLDER,
        fileNamePrefix=f'patches_{env}',
        fileFormat='CSV',
    )
    task.start()
    export_tasks[env] = task
    print(f"  Export started: patches_{env}.csv  id={task.id}")

# Poll all exports
print("\nWaiting for GEE exports…")
POLL_INTERVAL = 20
pending = dict(export_tasks)
while pending:
    time.sleep(POLL_INTERVAL)
    done = []
    for env, task in pending.items():
        state = task.status()['state']
        print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {env}: {state}")
        if state == 'COMPLETED':
            print(f"  ✓ {env}")
            done.append(env)
        elif state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(
                f"{env} export failed: {task.status().get('error_message', state)}"
            )
    for k in done:
        del pending[k]

print("\nAll CSVs exported. Downloading from Drive…")

# =============================================================================
# PHASE B — Download CSVs from Drive, convert to numpy, reupload as .npy
# =============================================================================

from googleapiclient.discovery import build
from googleapiclient.http      import MediaIoBaseDownload, MediaIoBaseUpload
from google.auth               import default as google_auth_default
from google.oauth2.credentials import Credentials

creds = Credentials.from_authorized_user_file('drive_token.json')
drive = build('drive', 'v3', credentials=creds)

def find_drive_file(name):
    """Return the Drive file id for the most recent file matching name."""
    results = drive.files().list(
        q=f"name='{name}' and trashed=false",
        orderBy='createdTime desc',
        pageSize=1,
        fields='files(id, name)'
    ).execute()
    files = results.get('files', [])
    if not files:
        raise FileNotFoundError(f"'{name}' not found in Drive")
    return files[0]['id']

def download_bytes(file_id):
    """Download a Drive file and return raw bytes."""
    req = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return buf.read()

def get_or_create_folder(name, parent_id=None):
    """Return Drive folder id, creating it if necessary."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    results = drive.files().list(q=q, fields='files(id)').execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        meta['parents'] = [parent_id]
    folder = drive.files().create(body=meta, fields='id').execute()
    return folder['id']

def upload_npy(array, filename, folder_id):
    """Upload a numpy array as .npy to a Drive folder."""
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype='application/octet-stream')
    drive.files().create(
        body={'name': filename, 'parents': [folder_id]},
        media_body=media,
        fields='id'
    ).execute()
    print(f"  Uploaded: {filename}  shape={array.shape}  dtype={array.dtype}")

# ── Find/create the npy subfolder in Drive ───────────────────────────────────
# Locate PlasticClassifier root folder
root_results = drive.files().list(
    q="name='PlasticClassifier' and mimeType='application/vnd.google-apps.folder' and trashed=false",
    fields='files(id)'
).execute()
root_id = root_results['files'][0]['id'] if root_results['files'] else None
npy_folder_id = get_or_create_folder('npy', parent_id=root_id)
print(f"Drive npy folder id: {npy_folder_id}")

# ── Download CSVs and parse into numpy ───────────────────────────────────────
import csv

X_all    = []   # embedding vectors:  (N, 64)  float32
y_all    = []   # labels:             (N,)     int8
meta_all = []   # metadata dicts:     (N,)     object

B_COLS = None   # filled on first parse

for env in ['beach', 'river']: #, 'ocean'
    fname   = f'patches_{env}.csv'
    file_id = find_drive_file(fname)
    raw     = download_bytes(file_id).decode('utf-8')
    reader  = csv.DictReader(io.StringIO(raw))

    rows = list(reader)
    print(f"\n  {fname}: {len(rows)} rows")

    for row in rows:
        if B_COLS is None:
            # Identify embedding band columns: named B0, B1 … B63
            B_COLS = sorted(
                [k for k in row.keys() if k.startswith('A') and k[1:].isdigit()],
                key=lambda k: int(k[1:])
            )
            print(f"  Embedding columns: {B_COLS[0]} … {B_COLS[-1]}  ({len(B_COLS)} dims)")

        try:
            vec = np.array([float(row[b]) for b in B_COLS], dtype=np.float32)
        except (ValueError, KeyError):
            continue  # skip rows with missing band values

        X_all.append(vec)
        y_all.append(int(row['label']))
        meta_all.append({
            'site_name':   row.get('site_name', ''),
            'environment': row.get('environment', ''),
            'site_lon':    float(row.get('site_lon', 0)),
            'site_lat':    float(row.get('site_lat', 0)),
        })

X = np.stack(X_all, axis=0)          # (N, 64)  float32
y = np.array(y_all, dtype=np.int8)   # (N,)     int8
meta = np.array(meta_all)            # (N,)     object

print(f"\nFull dataset: {X.shape[0]} pixels")
print(f"  Positive (plastic): {y.sum()}")
print(f"  Negative (clean):   {(y == 0).sum()}")
print(f"  Feature dims:       {X.shape[1]}")

# Per-environment arrays — useful for environment-stratified training
for env in ['beach', 'river']: #, 'ocean'
    idx = np.array([i for i, m in enumerate(meta) if m['environment'] == env])
    print(f"  {env:6s}: {len(idx)} pixels  "
          f"({y[idx].sum()} plastic / {(y[idx]==0).sum()} clean)")

# =============================================================================
# PHASE C — Save .npy files and upload to Drive
# =============================================================================

print("\nUploading .npy files to Drive…")
upload_npy(X,    'X_embeddings.npy', npy_folder_id)   # (N, 64) float32
upload_npy(y,    'y_labels.npy',     npy_folder_id)   # (N,)    int8
upload_npy(meta, 'meta.npy',         npy_folder_id)   # (N,)    object dicts

# Also save per-environment splits for convenience
for env in ['beach', 'river']: #, 'ocean'
    idx = np.array([i for i, m in enumerate(meta) if m['environment'] == env])
    if len(idx) == 0:
        continue
    upload_npy(X[idx], f'X_{env}.npy', npy_folder_id)
    upload_npy(y[idx], f'y_{env}.npy', npy_folder_id)

print(f"""
Download Complete ✓
Drive folder: PlasticClassifier/npy/
  X_embeddings.npy   shape ({X.shape[0]}, 64)  float32  — all environments
  y_labels.npy       shape ({y.shape[0]},)      int8     — 0=clean, 1=plastic
  meta.npy           shape ({meta.shape[0]},)   object   — site/env per pixel
  X_beach.npy / y_beach.npy
  X_river.npy / y_river.npy
  X_ocean.npy / y_ocean.npy

Local usage:
  import numpy as np
  X    = np.load('X_embeddings.npy')
  y    = np.load('y_labels.npy')
  meta = np.load('meta.npy', allow_pickle=True)

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
  rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5)
  rf.fit(X_tr, y_tr)
  print(rf.score(X_te, y_te))
""")
