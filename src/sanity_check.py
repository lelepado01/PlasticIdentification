import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
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
AREA = Area_3

def get_pixel_ranges(Area): 
    xm, yM, xM, ym = TOTAL_AREA
    x1, y1, x2, y2 = Area

    x1 = X_LEN * (x1 - xm) / (xM - xm)
    x2 = X_LEN * (x2 - xm) / (xM - xm)
    y1 = Y_LEN * (y1 - ym) / (yM - ym)
    y2 = Y_LEN * (y2 - ym) / (yM - ym)

    return [int(i) for i in [x1, y2, x2, y1]]


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_tif(path):
    img = rio.open_rasterio(path, chunks=True,lock=False)
    img = img.astype(np.float32)
    return img

def normalize(img, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanpercentile(img, 5)
    if vmax is None:
        vmax = np.nanpercentile(img, 95)
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)

def extract_date(fname):
    match = re.search(r"(20\d{2})[_-]?(0[1-9]|1[0-2])", fname)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"


files = sorted(glob.glob(os.path.join(input_dir, f"*.tif")))
del files[12]

m = [0.0, 0.0, 0.0]
M = [25056.0, 22960.0, 21472.0]

ch1 = []
ch2 = []
ch3 = []
for f in tqdm(files):
    img_norm = read_tif(f)[:3]
    img_norm = np.transpose(np.array(img_norm), [1, 2, 0])
    img_norm = normalize(img_norm)
    img_norm = np.nan_to_num(img_norm, 0)
    if img_norm.sum() == 0: 
        continue

    x1, y1, x2, y2 = get_pixel_ranges(AREA)
    img_norm = img_norm[y1:y2, x1:x2, :]

    ch1.append(img_norm[:, :, 0].mean())
    ch2.append(img_norm[:, :, 1].mean())
    ch3.append(img_norm[:, :, 2].mean())

plt.plot(ch1)
plt.plot(ch2)
plt.plot(ch3)
plt.show()