import ee
import requests
import json
import math
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
import io

ee.Authenticate()
ee.Initialize(project='plastic-483715')

OUT_DIR  = Path('swaths')
OUT_DIR.mkdir(exist_ok=True)

YEAR      = '2022'
SWATH_M   = 10_000   # 10km x 10km
PX        = 1000     # 1000x1000 pixels = 10m/px

# =============================================================================
# GPGP grid definition
#
# Covers the full documented accumulation zone from:
#   Lebreton et al. 2018 — "Evidence that the Great Pacific Garbage Patch
#   is rapidly accumulating plastic"
#
# Lon 130W–160W  x  Lat 27N–43N
# At 10km spacing: ~273 cols x 178 rows = ~48,594 tiles
# =============================================================================

LON_MIN, LON_MAX = -140.0, -130.0 #-160.0, -130.0
LAT_MIN, LAT_MAX =   27.0,   33.0 # 27.0,   43.0

MEAN_LAT  = (LAT_MIN + LAT_MAX) / 2
STEP_LAT  = SWATH_M / 111_000                              # deg per 10km NS
STEP_LON  = SWATH_M / (111_000 * math.cos(math.radians(MEAN_LAT)))  # deg per 10km EW

lons = [LON_MIN + i * STEP_LON
        for i in range(math.ceil((LON_MAX - LON_MIN) / STEP_LON))]
lats = [LAT_MIN + j * STEP_LAT
        for j in range(math.ceil((LAT_MAX - LAT_MIN) / STEP_LAT))]

# Build full site list: (col_idx, row_idx, lon, lat, name)
SITES = [
    (ci, ri, lon, lat, f'gpgp_{ci:03d}_{ri:03d}')
    for ri, lat in enumerate(lats)
    for ci, lon in enumerate(lons)
]

print(f"GPGP grid: {len(lons)} cols x {len(lats)} rows = {len(SITES):,} tiles")
print(f"Step: {STEP_LON:.4f}° lon  {STEP_LAT:.4f}° lat  (~10km each)")
print(f"Coverage: {LON_MIN}°–{LON_MAX}° lon,  {LAT_MIN}°–{LAT_MAX}° lat")

already = len(list(OUT_DIR.glob('*.png')))
print(f"Already downloaded: {already}  Remaining: {len(SITES)-already}\n")

# =============================================================================
# Fetch one swath — called from thread pool
# =============================================================================

def fetch_swath(ci, ri, lon, lat, name):
    png_path  = OUT_DIR / f'{name}.png'
    json_path = OUT_DIR / f'{name}.json'

    # Resume: skip if both files already exist and PNG is valid
    if png_path.exists() and json_path.exists() and png_path.stat().st_size > 1000:
        return 'skip', name

    half = SWATH_M / 2
    geom = ee.Geometry.Point([lon, lat]).buffer(half).bounds()

    coords  = geom.coordinates().getInfo()[0]
    lons_c  = [c[0] for c in coords]
    lats_c  = [c[1] for c in coords]
    lon_min, lon_max = min(lons_c), max(lons_c)
    lat_min, lat_max = min(lats_c), max(lats_c)

    candidate = None
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(geom)
                .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                .select(['B4', 'B3', 'B2']))
    if col.size().getInfo() > 0:
        candidate = col

    if candidate is None:
        return 'no_data', name

    s2 = (candidate.median()
                   .clip(geom)
                   .clamp(0, 3000)
                   .divide(3000)
                   .multiply(255)
                   .toUint8())

    url = s2.getThumbURL({
        'region':     geom,
        'dimensions': f'{PX}x{PX}',
        'format':     'png',
        'bands':      ['B4', 'B3', 'B2'],
    })

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                return 'http_error', name
            time.sleep(5 * (attempt + 1))

    img = Image.open(io.BytesIO(resp.content)).convert('RGB')
    w, h = img.size
    img.save(png_path)

    meta = {
        'name':      name,
        'col':       ci,
        'row':       ri,
        'lon':       lon,
        'lat':       lat,
        'lon_min':   lon_min,
        'lon_max':   lon_max,
        'lat_min':   lat_min,
        'lat_max':   lat_max,
        'width_px':  w,
        'height_px': h,
        'swath_m':   SWATH_M,
        'pixel_m':   SWATH_M / w,
    }
    json_path.write_text(json.dumps(meta, indent=2))
    return 'ok', name

# =============================================================================
# Parallel download loop
#
# GEE rate-limits at ~100 concurrent requests. We use 12 threads which keeps
# us well under the limit while still being ~10x faster than sequential.
# Progress is printed every 50 tiles.
# =============================================================================

WORKERS    = 12
BATCH_SIZE = 100   # print progress every N tiles

counts = {'ok': 0, 'skip': 0, 'no_data': 0, 'http_error': 0}
total  = len(SITES)
done   = 0

print(f"Starting parallel download ({WORKERS} workers)...\n")
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {
        pool.submit(fetch_swath, ci, ri, lon, lat, name): name
        for ci, ri, lon, lat, name in SITES
    }

    for fut in concurrent.futures.as_completed(futures):
        status, name = fut.result()
        counts[status] += 1
        done += 1

        if done % BATCH_SIZE == 0 or done == total:
            elapsed  = time.time() - t0
            rate     = done / elapsed
            eta_s    = (total - done) / rate if rate > 0 else 0
            eta_min  = eta_s / 60
            print(
                f"  [{done:>6}/{total}]  "
                f"ok={counts['ok']}  skip={counts['skip']}  "
                f"no_data={counts['no_data']}  err={counts['http_error']}  "
                f"rate={rate:.1f}/s  ETA={eta_min:.0f}min"
            )

elapsed_min = (time.time() - t0) / 60
print(f"""
Done in {elapsed_min:.1f} min
  Downloaded : {counts['ok']:>6}
  Skipped    : {counts['skip']:>6}  (already on disk)
  No S2 data : {counts['no_data']:>6}  (open ocean gaps — normal for GPGP)
  Errors     : {counts['http_error']:>6}

Total PNGs in {OUT_DIR}/: {len(list(OUT_DIR.glob('*.png')))}
""")