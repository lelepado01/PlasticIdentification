
"""
browse_swaths.py — Grid browser for downloaded swath PNGs

Displays a NxN grid of swath thumbnails. Arrow keys / buttons to page through.
Click any tile to zoom into it full-screen.

Usage:
  python pipeline/plot_swath_joined.py              # all swaths in swaths/
  python pipeline/plot_swath_joined.py 8            # 8x8 grid (default: 6)
  python pipeline/plot_swath_joined.py 6 gpgp_001  # start at tiles matching prefix
"""

import sys
import math
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

SWATHS_DIR = Path('swaths')
GRID_N     = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # NxN grid
PREFIX     = sys.argv[2]       if len(sys.argv) > 2 else ''

# ── Discover all PNGs ────────────────────────────────────────────────────────
all_pngs = sorted(SWATHS_DIR.glob('*.png'))
if PREFIX:
    all_pngs = [p for p in all_pngs if p.stem.startswith(PREFIX)]

if not all_pngs:
    print(f"No PNGs found in {SWATHS_DIR}/")
    sys.exit(1)

TILES_PER_PAGE = GRID_N * GRID_N
N_PAGES        = math.ceil(len(all_pngs) / TILES_PER_PAGE)
print(f"Found {len(all_pngs)} swaths  |  {GRID_N}x{GRID_N} grid  |  {N_PAGES} pages")

# ── State ─────────────────────────────────────────────────────────────────────
state = {'page': 0}

# ── Figure setup ─────────────────────────────────────────────────────────────
FIG_SIZE = min(14, GRID_N * 2.2)
fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), facecolor='#0d0d1a')
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07,
                    hspace=0.04, wspace=0.04)

# Button axes
ax_prev = fig.add_axes([0.02,  0.01, 0.12, 0.04])
ax_next = fig.add_axes([0.16,  0.01, 0.12, 0.04])
ax_info = fig.add_axes([0.30,  0.01, 0.40, 0.04])
ax_info.axis('off')
info_text = ax_info.text(0.5, 0.5, '', ha='center', va='center',
                          color='white', fontsize=9,
                          transform=ax_info.transAxes)

from matplotlib.widgets import Button
btn_style  = dict(color='#1e2a3a', hovercolor='#2e4060')
btn_prev   = Button(ax_prev, '◀  Prev', **btn_style)
btn_next   = Button(ax_next, 'Next  ▶', **btn_style)
btn_prev.label.set_color('white')
btn_next.label.set_color('white')

# Grid axes — created once, reused across pages
grid_axes = []
gs = gridspec.GridSpec(GRID_N, GRID_N,
                       figure=fig,
                       left=0.01, right=0.99,
                       top=0.93, bottom=0.07,
                       hspace=0.04, wspace=0.04)
for r in range(GRID_N):
    for c in range(GRID_N):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor('#0d0d1a')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        grid_axes.append(ax)

# ── Load and downsample a PNG to a small thumbnail ───────────────────────────
THUMB_PX = 128   # display at 128x128 regardless of source size

def load_thumb(path):
    try:
        img = Image.open(path).convert('RGB').resize((THUMB_PX, THUMB_PX), Image.BILINEAR)
        return np.array(img)
    except Exception:
        return np.zeros((THUMB_PX, THUMB_PX, 3), dtype=np.uint8)

def load_meta(path):
    jpath = path.with_suffix('.json')
    if jpath.exists():
        return json.loads(jpath.read_text())
    return {}

# ── Render one page ───────────────────────────────────────────────────────────
def render_page(page):
    start = page * TILES_PER_PAGE
    tiles = all_pngs[start : start + TILES_PER_PAGE]

    for i, ax in enumerate(grid_axes):
        ax.cla()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#0d0d1a')
        for sp in ax.spines.values():
            sp.set_visible(False)

        if i >= len(tiles):
            continue

        path  = tiles[i]
        thumb = load_thumb(path)
        ax.imshow(thumb, aspect='auto')

        # Label: stem name, abbreviated
        name = path.stem
        short = name.replace('gpgp_', '')   # trim prefix for compactness
        ax.set_xlabel(short, fontsize=5, color='#aaaacc',
                      labelpad=1, ha='center')

    n_this = len(tiles)
    info_text.set_text(
        f"Page {page+1} / {N_PAGES}   |   "
        f"Tiles {start+1}–{start+n_this} of {len(all_pngs)}   |   "
        f"Click tile to zoom   |   ← → arrow keys to page"
    )
    fig.suptitle('GPGP Swath Browser', color='white', fontsize=11, y=0.975)
    fig.canvas.draw_idle()

# ── Zoom into a single tile ───────────────────────────────────────────────────
def zoom_tile(path):
    meta = load_meta(path)
    img  = np.array(Image.open(path).convert('RGB'))

    zfig, zax = plt.subplots(figsize=(7, 7), facecolor='#0d0d1a')
    zfig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
    zax.imshow(img)
    zax.set_facecolor('#0d0d1a')
    zax.tick_params(colors='#aaaacc', labelsize=7)

    if meta:
        # Geographic axis ticks
        w, h = meta['width_px'], meta['height_px']
        xt = np.linspace(0, w, 5)
        xl = [f"{meta['lon_min'] + x/w*(meta['lon_max']-meta['lon_min']):.3f}°"
              for x in xt]
        yt = np.linspace(0, h, 5)
        yl = [f"{meta['lat_max'] - y/h*(meta['lat_max']-meta['lat_min']):.3f}°"
              for y in yt]
        zax.set_xticks(xt); zax.set_xticklabels(xl, color='#aaaacc', fontsize=7)
        zax.set_yticks(yt); zax.set_yticklabels(yl, color='#aaaacc', fontsize=7)

        title = (f"{path.stem}   "
                 f"({meta.get('lon', 0):.4f}°, {meta.get('lat', 0):.4f}°)   "
                 f"{meta.get('swath_m', 0)/1000:.0f}km × "
                 f"{meta.get('swath_m', 0)/1000:.0f}km   "
                 f"~{meta.get('pixel_m', 0):.0f} m/px")
    else:
        title = path.stem

    zax.set_title(title, color='white', fontsize=8, pad=6)
    for sp in zax.spines.values():
        sp.set_color('#333355')
    plt.show()

# ── Event handlers ────────────────────────────────────────────────────────────
def go_next(event=None):
    if state['page'] < N_PAGES - 1:
        state['page'] += 1
        render_page(state['page'])

def go_prev(event=None):
    if state['page'] > 0:
        state['page'] -= 1
        render_page(state['page'])

def on_key(event):
    if   event.key in ('right', 'n', 'pagedown'): go_next()
    elif event.key in ('left',  'p', 'pageup'):   go_prev()
    elif event.key == 'q':                         plt.close('all')

def on_click(event):
    # Identify which grid cell was clicked
    for i, ax in enumerate(grid_axes):
        if event.inaxes == ax:
            page_start = state['page'] * TILES_PER_PAGE
            tile_idx   = page_start + i
            if tile_idx < len(all_pngs):
                zoom_tile(all_pngs[tile_idx])
            return

btn_next.on_clicked(go_next)
btn_prev.on_clicked(go_prev)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

# ── Initial render ────────────────────────────────────────────────────────────
render_page(0)
plt.show()