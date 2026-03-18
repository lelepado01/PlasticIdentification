"""
visualize_patches.py

Fetches small S2 RGB thumbnail chips from GEE for every labeled site
and saves them as PNGs, grouped into a contact sheet per environment.

Outputs:
  patch_previews/beach.png
  patch_previews/river.png
  patch_previews/ocean.png
"""

import ee
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ee.Authenticate()
ee.Initialize(project='plastic-483715')

OUT_DIR = Path('patch_previews')
OUT_DIR.mkdir(exist_ok=True)

YEAR = '2022'

# Same labeled sites as the dataset — (lon, lat, buffer_m, label, env, name)
LABELED_REGIONS = [
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

# =============================================================================
# Fetch one RGB chip per site via getThumbURL
#
# getThumbURL renders a small image server-side and returns a PNG URL —
# no export task needed, no Drive required, result is immediate.
#
# Stretch: 0-3000 reflectance -> 0-255 (coastal/water scenes)
# Chip size: 128x128 pixels at 10m = ~1.28km square
# =============================================================================

def fetch_chip(lon, lat, buffer_m, label, name):
    """
    Returns an (H, W, 3) uint8 numpy array, or None on failure.
    """
    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

    # Try progressively looser filters until we get at least one scene.
    # Empty collections produce 0-band images that crash all downstream ops.
    candidate = None
    for (date_start, date_end, cloud_pct) in [
        (f'{YEAR}-01-01',   f'{YEAR}-12-31',   20),   # ideal: same year, low cloud
        ('2020-01-01',      '2023-12-31',       20),   # wider years, same cloud limit
        ('2020-01-01',      '2023-12-31',       50),   # accept cloudier scenes
        ('2018-01-01',      '2024-12-31',       80),   # last resort
    ]:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(geom)
                 .filterDate(date_start, date_end)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                 .select(['B4', 'B3', 'B2']))
        if col.size().getInfo() > 0:
            candidate = col
            break

    if candidate is None:
        print(f"    WARNING: no S2 scenes found for {name}, skipping.")
        return None

    s2 = candidate.median().clip(geom)

    # Stretch to 8-bit display range
    s2_display = (s2.clamp(0, 3000)
                    .divide(3000)
                    .multiply(255)
                    .toUint8())

    url = s2_display.getThumbURL({
        'region':      geom,
        'dimensions':  128,          # 128x128 px output
        'format':      'png',
        'min':         0,
        'max':         255,
        'bands':       ['B4', 'B3', 'B2'],
    })

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Parse PNG bytes -> numpy array without saving to disk
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return np.array(img)

    except Exception as e:
        print(f"    WARNING: could not fetch {name}: {e}")
        return None


# =============================================================================
# Build contact sheet per environment
#
# Layout: two rows — top row = plastic (label=1), bottom row = clean (label=0)
# Each cell = one site thumbnail with site name and label colour border
# =============================================================================

BORDER_PX    = 4
LABEL_COLORS = {1: '#e74c3c', 0: '#2ecc71'}   # red = plastic, green = clean
LABEL_TEXT   = {1: 'PLASTIC', 0: 'CLEAN'}

def add_border(img, label):
    """Add a colored border around a (H, W, 3) chip."""
    color_hex = LABEL_COLORS[label].lstrip('#')
    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    h, w = img.shape[:2]
    bordered = np.full((h + 2*BORDER_PX, w + 2*BORDER_PX, 3),
                       color_rgb, dtype=np.uint8)
    bordered[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = img
    return bordered


for env in ['beach', 'river', 'ocean']:
    sites = [(lon, lat, buf, lbl, name)
             for lon, lat, buf, lbl, e, name in LABELED_REGIONS
             if e == env]

    plastic_sites = [(l, n, fetch_chip(lo, la, b, l, n))
                     for lo, la, b, l, n in sites if l == 1]
    clean_sites   = [(l, n, fetch_chip(lo, la, b, l, n))
                     for lo, la, b, l, n in sites if l == 0]

    # Drop failed fetches
    plastic_chips = [(lbl, name, chip) for lbl, name, chip in plastic_sites if chip is not None]
    clean_chips   = [(lbl, name, chip) for lbl, name, chip in clean_sites   if chip is not None]

    n_cols = max(len(plastic_chips), len(clean_chips))
    if n_cols == 0:
        print(f"  No chips for {env}, skipping.")
        continue

    CHIP_SIZE = 128 + 2 * BORDER_PX   # chip + border
    FIG_W     = n_cols * (CHIP_SIZE / 72) + 1.5
    FIG_H     = 2     * (CHIP_SIZE / 72) + 1.2

    fig, axes = plt.subplots(2, n_cols,
                             figsize=(FIG_W, FIG_H),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.patch.set_facecolor('#1a1a2e')

    row_data = [plastic_chips, clean_chips]
    for row_idx, chips in enumerate(row_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#1a1a2e')
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx < len(chips):
                lbl, name, chip = chips[col_idx]
                bordered = add_border(chip, lbl)
                ax.imshow(bordered)
                # Site name below the chip
                display_name = name.replace('_', '\n')
                ax.set_xlabel(display_name, fontsize=5.5,
                              color='white', labelpad=2)
            else:
                # Empty cell — keep consistent grid
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # Row labels on left
    for row_idx, (lbl, chips) in enumerate(zip([1, 0], row_data)):
        if len(chips) > 0:
            axes[row_idx, 0].set_ylabel(
                LABEL_TEXT[lbl], fontsize=9, fontweight='bold',
                color=LABEL_COLORS[lbl], rotation=90, labelpad=6
            )

    # Legend patches
    legend_elements = [
        mpatches.Patch(color=LABEL_COLORS[1], label='Plastic (positive)'),
        mpatches.Patch(color=LABEL_COLORS[0], label='Clean (negative)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=8, facecolor='#1a1a2e',
               labelcolor='white', framealpha=0.5,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f'Labeled patches — {env.capitalize()}  ({YEAR} S2 median)',
                 color='white', fontsize=11, y=0.98)

    out_path = OUT_DIR / f'{env}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved -> {out_path}  "
          f"({len(plastic_chips)} plastic / {len(clean_chips)} clean chips)")

print(f"\nDone. Contact sheets saved to {OUT_DIR}/")