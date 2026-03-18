import glob
import os
import re
import numpy as np
import imageio
from tqdm import tqdm
import rioxarray as rio
from PIL import Image

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
input_dir = "GEE_Sentinel2_Monthly"      # folder with GeoTIFFs
output_dir = "gifs"
os.makedirs(output_dir, exist_ok=True)

bands = ["B2", "B3", "B4", "B8"]

TOTAL_AREA = [-76.870946, 17.924683, -76.716177, 18.0057]
X_LEN = 1724
Y_LEN = 903

Area_1 = [-76.850363, 17.992451, -76.834513, 18.003829] 
Area_2 = [-76.817179, 17.963019, -76.800637, 17.974251]
Area_3 = [-76.78637, 17.962443, -76.773754, 17.965535]
Area_4 = [-76.760429, 17.965106, -76.754388, 17.968588]

def get_pixel_ranges(Area): 
    xm, yM, xM, ym = TOTAL_AREA
    x1, y1, x2, y2 = Area

    x1 = X_LEN * (x1 - xm) / (xM - xm)
    x2 = X_LEN * (x2 - xm) / (xM - xm)
    y1 = Y_LEN * (y1 - ym) / (yM - ym)
    y2 = Y_LEN * (y2 - ym) / (yM - ym)

    return [int(i) for i in [x1, y1, x2, y2]]


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_tif(path):
    ds = rio.open_rasterio(path, chunks=True,lock=False)
    # with rasterio.open(path) as src:
    img = ds.astype(np.float32)
    # x1, y1, x2, y2 = get_pixel_ranges(Area_1)
    # img = img[:, y1:y2, x1:x2]
    # img[img == ds.nodata] = np.nan
    return img

def normalize(img, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanpercentile(img, 2)
    if vmax is None:
        vmax = np.nanpercentile(img, 98)
    img = np.clip(img, vmin, vmax)
    return ((img - vmin) / (vmax - vmin) * 255)#.astype(np.uint8)

def extract_date(fname):
    match = re.search(r"(20\d{2})[_-]?(0[1-9]|1[0-2])", fname)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"

# -----------------------------
# PROCESS EACH BAND
# -----------------------------
# for i, band in enumerate(bands):

# print(f"\nProcessing {band}...")

files = sorted(glob.glob(os.path.join(input_dir, f"*.tif")))
f = files[0]
img = read_tif(f)[:3]
img_norm = normalize(img) / 255.0
img_norm = np.transpose(np.array(img_norm), [1, 2, 0])
img_norm = Image.fromarray((img_norm * 255).astype(np.uint8))
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img_norm)

areas = {
    "Area 1": Area_1,
    "Area 2": Area_2,
    "Area 3": Area_3,
    "Area 4": Area_4,
}

for name, area in areas.items():
    x1, y1, x2, y2 = get_pixel_ranges(area)

    # Ensure correct ordering
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    width = x_max - x_min
    height = y_max - y_min

    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)
    ax.text(
        x_min,
        y_min - 5,
        name,
        color="red",
        fontsize=9,
        weight="bold"
    )

ax.set_axis_off()
plt.tight_layout()
plt.savefig("areas.png")
plt.show()


# frames = []
# for f in tqdm(files):
#     img = read_tif(f)[:3]
#     img_norm = normalize(img) / 255.0
#     img_norm = np.transpose(np.array(img_norm), [1, 2, 0])
#     img_norm = Image.fromarray((img_norm * 255).astype(np.uint8))
#     frames.append(img_norm)

# gif_path = os.path.join(output_dir, "rgb.gif")#f"{band}.gif")

# imageio.mimsave(
#     gif_path,
#     frames,
#     fps=1
# )
