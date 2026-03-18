
import matplotlib.pyplot as plt
import rasterio

rgb = rasterio.open("overview_rgb_10km.tif")
rgb = rgb.read()
rgb = rgb.transpose(1, 2, 0)
plt.imshow(rgb, alpha=0.2)


with rasterio.open("plastic_patches_s2_cosine.tif") as src:
    data = src.read()[0]
    plt.imshow(data, alpha=0.5)
    plt.show()

