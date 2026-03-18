
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rio

ds = rio.open_rasterio('higher_res/5ebbfdfa80832d00073367cc.tif', chunks=True,lock=False)
ds = np.transpose(np.array(ds), [1, 2, 0])
print(ds.shape)

# for x in range(4): 
#     for y in range(9): 
#         plt.imshow(ds[10000 * x:10000 * (x+1), 10000 * y:10000 * (y+1), 0])
#         plt.savefig(f"high_res_{x}_{y}.pdf")
#         # plt.show()
#         plt.close()


# AREA 1
# x = 2
# y = 4
# ds = ds[10000 * x:10000 * (x+1), 10000 * y:10000 * (y+1)]
# ds = ds[2500:3500, :1000]
# AREA 2
x = 2
y = 5
ds = ds[10000 * x:10000 * (x+1), 10000 * y:10000 * (y+1)]
ds = ds[9000:10000, 3200:4000]

plt.imshow(ds)
plt.savefig(f"high_res_{x}_{y}_subs.pdf")
plt.show()
plt.close()