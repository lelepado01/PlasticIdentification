import ee
import geemap

# Initialize Earth Engine
ee.Authenticate(force=True)
ee.Initialize(project='plastic-483715')

# -----------------------------
# USER PARAMETERS
# -----------------------------

# Bounding box (min_lon, min_lat, max_lon, max_lat)
roi = ee.Geometry.Rectangle([-76.870946, 17.924683, -76.716177, 18.0057])

start_year = 2021
end_year = 2025
cloud_percentage = 10  # max cloud cover %
bands_10m = ["B2", "B3", "B4", "B8"]

# -----------------------------
# LOOP: YEAR → MONTH
# -----------------------------
for year in range(start_year, end_year + 1):
    for month in range(1, 13):

        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage))
            .select(bands_10m)
        )

        # Skip empty months
        if collection.size().getInfo() == 0:
            print(f"No images for {year}-{month:02d}, skipping.")
            continue

        image = collection.median().clip(roi)

        export_name = f"S2_10m_{year}_{month:02d}"

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=export_name,
            folder="GEE_Sentinel2_Monthly",
            fileNamePrefix=export_name,
            region=roi,
            scale=10,
            crs="EPSG:4326",
            maxPixels=1e13
        )

        task.start()
        print(f"Started export: {export_name}")

print("All monthly exports submitted.")