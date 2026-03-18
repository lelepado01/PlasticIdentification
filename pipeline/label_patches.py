"""
label_patches.py — Interactive patch labeling tool

Left-click   → mark patch as PLASTIC  (red)
Right-click  → mark patch as CLEAN    (green)
Middle-click → remove label from patch
S key        → save labels to CSV
N key        → next swath
P key        → previous swath
Z key        → undo last label
Q key        → quit and save

Usage:
  python label_patches.py                    # labels all swaths in swaths/
  python label_patches.py kingston_interceptor jakarta_bay   # specific swaths

Output:
  labels.csv  — one row per labeled patch:
    site_name, lon, lat, patch_lon, patch_lat, label (1=plastic, 0=clean)
"""

import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from pathlib import Path

SWATHS_DIR  = Path('swaths')
OUTPUT_CSV  = Path('labels.csv')
PATCH_M     = 10     # snap grid = 10m patches

# Colors
COL_PLASTIC = '#e74c3c'   # red
COL_CLEAN   = '#2ecc71'   # green
COL_GRID    = '#ffffff'   # faint white grid

# =============================================================================
# Helpers
# =============================================================================

def load_swath(name):
    png  = SWATHS_DIR / f'{name}.png'
    meta = SWATHS_DIR / f'{name}.json'
    if not png.exists() or not meta.exists():
        raise FileNotFoundError(f"Missing {png} or {meta}")
    img  = plt.imread(str(png))
    info = json.loads(meta.read_text())
    return img, info


def pixel_to_geo(px, py, info):
    """Convert (col, row) pixel click to (lon, lat) patch centre."""
    lon = info['lon_min'] + (px + 0.5) / info['width_px']  * (info['lon_max'] - info['lon_min'])
    lat = info['lat_max'] - (py + 0.5) / info['height_px'] * (info['lat_max'] - info['lat_min'])
    return round(lon, 7), round(lat, 7)


def geo_to_pixel(lon, lat, info):
    """Convert (lon, lat) back to (col, row) pixel for drawing."""
    px = (lon - info['lon_min']) / (info['lon_max'] - info['lon_min']) * info['width_px']
    py = (info['lat_max'] - lat) / (info['lat_max'] - info['lat_min']) * info['height_px']
    return px, py


def snap_to_grid(px, py, info):
    """
    Snap a click to the nearest 10m patch grid cell.
    Returns the grid-aligned pixel (top-left corner col, row) and
    the geographic centre of that cell.
    """
    # How many pixels wide is one 10m patch?
    px_per_patch = PATCH_M / info['pixel_m']

    grid_col = int(px / px_per_patch)
    grid_row = int(py / px_per_patch)

    # Top-left pixel of this grid cell
    tl_px = grid_col * px_per_patch
    tl_py = grid_row * px_per_patch

    # Centre of cell in pixel space
    cx_px = tl_px + px_per_patch / 2
    cy_py = tl_py + px_per_patch / 2

    lon, lat = pixel_to_geo(cx_px, cy_py, info)
    return (grid_col, grid_row), lon, lat, px_per_patch


# =============================================================================
# Labeler state
# =============================================================================

class Labeler:
    def __init__(self, swath_names):
        self.swath_names = swath_names
        self.idx         = 0
        self.history     = []   # for undo: list of (site, grid_key)

        # labels[(site_name, grid_col, grid_row)] = 0 or 1
        self.labels = {}

        # Load existing CSV if present
        if OUTPUT_CSV.exists():
            with open(OUTPUT_CSV) as f:
                for row in csv.DictReader(f):
                    key = (row['site_name'], int(row['grid_col']), int(row['grid_row']))
                    self.labels[key] = int(row['label'])
            print(f"Loaded {len(self.labels)} existing labels from {OUTPUT_CSV}")

        self._build_figure()
        self._load_current()

    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#1a1a2e')
        plt.subplots_adjust(bottom=0.12)

        # Buttons
        ax_save  = plt.axes([0.15, 0.02, 0.12, 0.05])
        ax_prev  = plt.axes([0.30, 0.02, 0.12, 0.05])
        ax_next  = plt.axes([0.45, 0.02, 0.12, 0.05])
        ax_undo  = plt.axes([0.60, 0.02, 0.12, 0.05])
        ax_clear = plt.axes([0.75, 0.02, 0.12, 0.05])

        btn_style = dict(color='#2c3e50', hovercolor='#34495e')
        self.btn_save  = Button(ax_save,  'Save (S)',  **btn_style)
        self.btn_prev  = Button(ax_prev,  'Prev (P)',  **btn_style)
        self.btn_next  = Button(ax_next,  'Next (N)',  **btn_style)
        self.btn_undo  = Button(ax_undo,  'Undo (Z)',  **btn_style)
        self.btn_clear = Button(ax_clear, 'Clear',     **btn_style)

        for btn in [self.btn_save, self.btn_prev, self.btn_next,
                    self.btn_undo, self.btn_clear]:
            btn.label.set_color('white')

        self.btn_save.on_clicked(lambda _: self.save())
        self.btn_prev.on_clicked(lambda _: self.prev())
        self.btn_next.on_clicked(lambda _: self.next())
        self.btn_undo.on_clicked(lambda _: self.undo())
        self.btn_clear.on_clicked(lambda _: self.clear_current())

        self.fig.canvas.mpl_connect('button_press_event',  self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',     self._on_key)

    def _load_current(self):
        name     = self.swath_names[self.idx]
        img, info = load_swath(name)
        self.current_img  = img
        self.current_info = info
        self.current_name = name
        self._redraw()

    def _redraw(self):
        self.ax.cla()
        self.ax.set_facecolor('#1a1a2e')
        info = self.current_info
        name = self.current_name

        self.ax.imshow(self.current_img, extent=[0, info['width_px'], info['height_px'], 0])

        # Draw faint 10m grid
        px_per_patch = PATCH_M / info['pixel_m']
        for x in np.arange(0, info['width_px'], px_per_patch):
            self.ax.axvline(x, color=COL_GRID, lw=0.3, alpha=0.25)
        for y in np.arange(0, info['height_px'], px_per_patch):
            self.ax.axhline(y, color=COL_GRID, lw=0.3, alpha=0.25)

        # Draw existing labels for this swath
        for (site, gc, gr), lbl in self.labels.items():
            if site != name:
                continue
            tl_px = gc * px_per_patch
            tl_py = gr * px_per_patch
            color = COL_PLASTIC if lbl == 1 else COL_CLEAN
            rect  = mpatches.Rectangle(
                (tl_px, tl_py), px_per_patch, px_per_patch,
                linewidth=1.2, edgecolor=color,
                facecolor=color, alpha=0.45
            )
            self.ax.add_patch(rect)

        # Axis labels in geographic coords
        n_ticks = 5
        xt = np.linspace(0, info['width_px'], n_ticks)
        xl = [f"{info['lon_min'] + x/info['width_px']*(info['lon_max']-info['lon_min']):.4f}"
              for x in xt]
        yt = np.linspace(0, info['height_px'], n_ticks)
        yl = [f"{info['lat_max'] - y/info['height_px']*(info['lat_max']-info['lat_min']):.4f}"
              for y in yt]

        self.ax.set_xticks(xt); self.ax.set_xticklabels(xl, color='white', fontsize=7)
        self.ax.set_yticks(yt); self.ax.set_yticklabels(yl, color='white', fontsize=7)
        self.ax.tick_params(colors='white')

        # Count labels for this swath
        site_labels = {k: v for k, v in self.labels.items() if k[0] == name}
        n_plastic = sum(1 for v in site_labels.values() if v == 1)
        n_clean   = sum(1 for v in site_labels.values() if v == 0)

        self.ax.set_title(
            f"{name}  [{self.idx+1}/{len(self.swath_names)}]\n"
            f"Left-click=plastic  Right-click=clean  Middle=remove\n"
            f"Plastic: {n_plastic}  Clean: {n_clean}  "
            f"(~{info['pixel_m']:.1f} m/px, grid={PATCH_M}m)",
            color='white', fontsize=9, pad=8
        )

        # Legend
        legend = [
            mpatches.Patch(color=COL_PLASTIC, alpha=0.7, label=f'Plastic ({n_plastic})'),
            mpatches.Patch(color=COL_CLEAN,   alpha=0.7, label=f'Clean ({n_clean})'),
        ]
        self.ax.legend(handles=legend, loc='lower right',
                       facecolor='#1a1a2e', labelcolor='white', fontsize=8)

        self.fig.canvas.draw_idle()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        (gc, gr), lon, lat, _ = snap_to_grid(
            event.xdata, event.ydata, self.current_info
        )
        key = (self.current_name, gc, gr)

        if event.button == 1:      # left  → plastic
            self.labels[key] = 1
            self.history.append(key)
            print(f"  PLASTIC  grid=({gc},{gr})  lon={lon:.6f}  lat={lat:.6f}")
        elif event.button == 3:    # right → clean
            self.labels[key] = 0
            self.history.append(key)
            print(f"  CLEAN    grid=({gc},{gr})  lon={lon:.6f}  lat={lat:.6f}")
        elif event.button == 2:    # middle → remove
            if key in self.labels:
                del self.labels[key]
                print(f"  REMOVED  grid=({gc},{gr})")

        self._redraw()

    def _on_key(self, event):
        if   event.key == 's': self.save()
        elif event.key == 'n': self.next()
        elif event.key == 'p': self.prev()
        elif event.key == 'z': self.undo()
        elif event.key == 'q': self.save(); plt.close('all')

    # ── actions ──────────────────────────────────────────────────────────────

    def next(self, *_):
        self.save()
        self.idx = (self.idx + 1) % len(self.swath_names)
        self._load_current()

    def prev(self, *_):
        self.save()
        self.idx = (self.idx - 1) % len(self.swath_names)
        self._load_current()

    def undo(self, *_):
        if self.history:
            key = self.history.pop()
            if key in self.labels:
                del self.labels[key]
                print(f"  UNDO  {key}")
            self._redraw()

    def clear_current(self, *_):
        name = self.current_name
        keys = [k for k in self.labels if k[0] == name]
        for k in keys:
            del self.labels[k]
        self.history = [h for h in self.history if h[0] != name]
        print(f"  Cleared all labels for {name}")
        self._redraw()

    def save(self, *_):
        fieldnames = ['site_name', 'grid_col', 'grid_row',
                      'patch_lon', 'patch_lat', 'label']
        rows = []
        for (site, gc, gr), lbl in sorted(self.labels.items()):
            info        = load_swath(site)[1]
            px_per_patch = PATCH_M / info['pixel_m']
            cx_px = gc * px_per_patch + px_per_patch / 2
            cy_py = gr * px_per_patch + px_per_patch / 2
            lon, lat = pixel_to_geo(cx_px, cy_py, info)
            rows.append({
                'site_name':  site,
                'grid_col':   gc,
                'grid_row':   gr,
                'patch_lon':  lon,
                'patch_lat':  lat,
                'label':      lbl,
            })

        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_plastic = sum(1 for r in rows if r['label'] == 1)
        n_clean   = sum(1 for r in rows if r['label'] == 0)
        print(f"  Saved {len(rows)} labels -> {OUTPUT_CSV}  "
              f"({n_plastic} plastic / {n_clean} clean)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    # Discover swaths: use CLI args or all PNGs in the swaths folder
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = sorted(p.stem for p in SWATHS_DIR.glob('*.png'))

    if not names:
        print(f"No swaths found in {SWATHS_DIR}/. Run download_swaths.py first.")
        sys.exit(1)

    print(f"Labeling {len(names)} swath(s): {', '.join(names)}")
    print("Left-click=plastic  Right-click=clean  Middle=remove")
    print("Keys: S=save  N=next  P=prev  Z=undo  Q=quit\n")

    labeler = Labeler(names)
    plt.show()
